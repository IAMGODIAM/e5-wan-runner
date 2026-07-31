"""E5 Sovereign Video Lane — Wan 2.2 on Modal. Two tiers, one body.

Tiers:
  small (default) — L4 24GB: TI2V-5B t2v/i2v, FastWan distill. Volume lane.
  big             — H100 80GB + 128Gi RAM: Wan2.2 A14B MoE (T2V/I2V) with the
                    LightX2V Lightning 4-step LoRA pair (high/low-noise experts).

Driven over HTTP/1.1: POST kick (gate-keyed) -> {call_id}; GET stat -> {done,result}.
Renders write MP4 + timing JSON to R2. Gate = Modal secret "wan-gate" (mirrored in
GH Actions secrets + treasurebox); Modal proxy auth is not used (needs dashboard-
minted tokens). Source-image URLs are short-lived presigned R2 URLs, never logged.
"""
import modal

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.7.1",
        "diffusers>=0.35.0",
        "transformers>=4.46.0",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "ftfy",
        "imageio",
        "imageio-ffmpeg",
        "boto3",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]",
        "peft",
    )
)
app = modal.App("e5-wan-video-modal", image=image)
vol = modal.Volume.from_name("e5-wan-hf", create_if_missing=True)

_COMMON = dict(
    timeout=3600,
    volumes={"/vol/hf": vol},
    secrets=[modal.Secret.from_name("r2-creds")],
)


def _render(
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    steps: int,
    guidance: float,
    seed: int,
    fps: int,
    out_key: str,
    model_id: str,
    lora_repo: str,
    lora_file: str,
    lora_scale: float,
    lora_file2: str,
    lora_scale2: float,
    flow_shift: float,
    image_url: str,
):
    import os, time, json, traceback

    os.environ.setdefault("HF_HOME", "/vol/hf")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    import boto3

    bucket = os.environ["R2_BUCKET"]
    s3c = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    err_key = out_key.rsplit("/", 1)[0] + "/error.txt"
    try:
        import torch
        from diffusers import WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan
        from diffusers.utils import export_to_video, load_image

        t0 = time.time()
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )
        pipe_cls = WanImageToVideoPipeline if image_url else WanPipeline
        pipe = pipe_cls.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        adapters, weights = [], []
        if lora_repo and lora_file:
            pipe.load_lora_weights(
                lora_repo, weight_name=lora_file, adapter_name="turbo"
            )
            adapters.append("turbo"); weights.append(float(lora_scale))
        if lora_repo and lora_file2:
            pipe.load_lora_weights(
                lora_repo, weight_name=lora_file2, adapter_name="turbo2",
                load_into_transformer_2=True,
            )
            adapters.append("turbo2"); weights.append(float(lora_scale2))
        if adapters:
            pipe.set_adapters(adapters, adapter_weights=weights)
        if float(flow_shift) > 0:
            from diffusers import UniPCMultistepScheduler

            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config, flow_shift=float(flow_shift)
            )
        pipe.enable_model_cpu_offload()
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        load_s = round(time.time() - t0, 1)
        vol.commit()  # persist freshly-downloaded weights for future warm runs

        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            generator=gen,
        )
        if image_url:
            call_kwargs["image"] = load_image(image_url)
        t1 = time.time()
        frames = pipe(**call_kwargs).frames[0]
        render_s = round(time.time() - t1, 1)

        out_path = "/tmp/out.mp4"
        export_to_video(frames, out_path, fps=int(fps))
        size = os.path.getsize(out_path)
        s3c.upload_file(out_path, bucket, out_key, ExtraArgs={"ContentType": "video/mp4"})

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        result = {
            "rc": 0,
            "out_key": out_key,
            "bytes": size,
            "gpu": gpu,
            "load_s": load_s,
            "render_s": render_s,
            "params": {
                "h": height, "w": width, "frames": num_frames, "steps": steps,
                "guidance": guidance, "seed": seed, "fps": fps, "model": model_id,
                "lora": (f"{lora_file}@{lora_scale}" if lora_file else ""),
                "lora2": (f"{lora_file2}@{lora_scale2}" if lora_file2 else ""),
                "flow_shift": flow_shift,
                "i2v": bool(image_url),
            },
        }
        s3c.put_object(
            Bucket=bucket,
            Key=out_key + ".json",
            Body=json.dumps(result, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        print(json.dumps(result))
        return result
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        try:
            s3c.put_object(Bucket=bucket, Key=err_key, Body=tb.encode("utf-8"))
        except Exception as e2:
            print("could not upload error.txt:", e2)
        return {"error": str(e), "error_key": err_key}


_DEFAULTS = dict(
    prompt="",
    negative_prompt=(
        "blurry, distorted, low quality, watermark, text, extra limbs, "
        "static image, jpeg artifacts"
    ),
    height=480,
    width=832,
    num_frames=81,
    steps=30,
    guidance=5.0,
    seed=42,
    fps=24,
    out_key="wan/proof/out.mp4",
    model_id=MODEL_ID,
    lora_repo="",
    lora_file="",
    lora_scale=1.0,
    lora_file2="",
    lora_scale2=1.0,
    flow_shift=0.0,
    image_url="",
)


def _merged(kw: dict) -> dict:
    p = dict(_DEFAULTS)
    p.update({k: v for k, v in (kw or {}).items() if k in _DEFAULTS})
    if not p["prompt"]:
        p["prompt"] = (
            "A majestic black dragon with gold-trimmed scales soars over a sunlit "
            "Miami skyline at golden hour, slow cinematic camera orbit, volumetric "
            "light, photoreal detail"
        )
    return p


@app.function(gpu="L4", **_COMMON)
def render_wan(**kw):
    return _render(**_merged(kw))


@app.function(gpu="H100", memory=131072, cpu=16.0, **_COMMON)
def render_wan_big(**kw):
    return _render(**_merged(kw))


@app.function(secrets=[modal.Secret.from_name("wan-gate")])
@modal.fastapi_endpoint(method="POST")
def kick(body: dict):
    import os

    payload = dict(body or {})
    if payload.pop("k", None) != os.environ.get("WAN_GATE"):
        return {"error": "unauthorized"}
    tier = payload.pop("tier", "small")
    fn = render_wan_big if tier == "big" else render_wan
    call = fn.spawn(**payload)
    return {"call_id": call.object_id, "tier": tier}


@app.function(secrets=[modal.Secret.from_name("wan-gate")])
@modal.fastapi_endpoint(method="GET")
def stat(call_id: str, k: str = ""):
    import os

    if k != os.environ.get("WAN_GATE"):
        return {"error": "unauthorized"}
    fc = modal.FunctionCall.from_id(call_id)
    try:
        res = fc.get(timeout=0)
        return {"done": True, "result": res}
    except TimeoutError:
        return {"done": False}
    except Exception as e:
        name = type(e).__name__
        if "Timeout" in name:
            return {"done": False}
        return {"done": True, "error": f"{name}: {e}"}
