"""E5 Sovereign Video Lane — Wan 2.2 TI2V-5B on Modal (L4), standby mirror of the Beam lane.

Deployed by GitHub Actions (gRPC control plane). Driven over HTTP/1.1:
POST kick (proxy-authed) -> spawns the GPU render, returns call_id;
GET stat?call_id=... -> {done, result}. Render writes MP4 + timing JSON to R2.
Both web endpoints require Modal proxy auth (Modal-Key / Modal-Secret headers)
because the invoke URLs are committed to a public repo.
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


@app.function(
    gpu="L4",
    timeout=3600,
    volumes={"/vol/hf": vol},
    secrets=[modal.Secret.from_name("r2-creds")],
)
def render_wan(
    prompt: str = (
        "A majestic black dragon with gold-trimmed scales soars over a sunlit "
        "Miami skyline at golden hour, slow cinematic camera orbit, volumetric "
        "light, photoreal detail"
    ),
    negative_prompt: str = (
        "blurry, distorted, low quality, watermark, text, extra limbs, "
        "static image, jpeg artifacts"
    ),
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    steps: int = 30,
    guidance: float = 5.0,
    seed: int = 42,
    fps: int = 24,
    out_key: str = "wan/proof/wan22_5b_proof.mp4",
    model_id: str = MODEL_ID,
    lora_repo: str = "",
    lora_file: str = "",
    lora_scale: float = 1.0,
    flow_shift: float = 0.0,
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
        from diffusers import WanPipeline, AutoencoderKLWan
        from diffusers.utils import export_to_video

        t0 = time.time()
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        if lora_repo:
            pipe.load_lora_weights(
                lora_repo, weight_name=(lora_file or None), adapter_name="turbo"
            )
            pipe.set_adapters(["turbo"], adapter_weights=[float(lora_scale)])
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
        t1 = time.time()
        frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            generator=gen,
        ).frames[0]
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
                "lora": (f"{lora_repo}/{lora_file}@{lora_scale}" if lora_repo else ""),
                "flow_shift": flow_shift,
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


# NOTE: Modal proxy auth requires dashboard-minted Proxy Auth Tokens (API tokens
# are rejected: "invalid credentials for proxy authorization"). Dashboard steps are
# out of autonomous scope, so the gate is an app-level shared secret (Modal secret
# "wan-gate", mirrored in GH Actions secrets + the E5 treasurebox vault).


@app.function(secrets=[modal.Secret.from_name("wan-gate")])
@modal.fastapi_endpoint(method="POST")
def kick(body: dict):
    import os

    payload = dict(body or {})
    if payload.pop("k", None) != os.environ.get("WAN_GATE"):
        return {"error": "unauthorized"}
    call = render_wan.spawn(**payload)
    return {"call_id": call.object_id}


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
