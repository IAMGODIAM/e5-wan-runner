"""E5 Sovereign Video Lane — FramePack (image-to-video) on Modal.

FramePack = lllyasviel's next-frame-prediction I2V, now a first-class diffusers
pipeline (HunyuanVideoFramepackPipeline). Runs headless on Modal, driven over
HTTP/1.1: POST kick (gate-keyed) -> {call_id}; GET stat -> {done,result}.
Renders write MP4 + timing JSON to R2 (bucket e5-agora at out_key); failures
write error.txt beside it. Reuses the wan lane's secrets (r2-creds, wan-gate)
and HF cache volume (e5-wan-hf). Mirror of modal_wan.py structure.

Weights (auto-cached to the volume on first run):
  transformer  = lllyasviel/FramePackI2V_HY        (the FramePack transformer)
  backbone     = hunyuanvideo-community/HunyuanVideo (VAE + text encoders)
  image encoder= lllyasviel/flux_redux_bfl          (SigLIP feature extractor)
"""
import modal

TRANSFORMER_ID = "lllyasviel/FramePackI2V_HY"
BACKBONE_ID = "hunyuanvideo-community/HunyuanVideo"
SIGLIP_ID = "lllyasviel/flux_redux_bfl"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.7.1",
        "diffusers>=0.34.0",
        "transformers>=4.46.0",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "ftfy",
        "pillow",
        "imageio",
        "imageio-ffmpeg",
        "boto3",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]",
    )
)
app = modal.App("e5-framepack-modal", image=image)
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
    image_url: str,
    sampling_type: str,
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
        from diffusers import (
            HunyuanVideoFramepackPipeline,
            HunyuanVideoFramepackTransformer3DModel,
        )
        from diffusers.utils import export_to_video, load_image
        from transformers import SiglipImageProcessor, SiglipVisionModel

        if not image_url:
            raise ValueError("FramePack is image-to-video: image_url is required")

        t0 = time.time()
        transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
            TRANSFORMER_ID, torch_dtype=torch.bfloat16
        )
        feature_extractor = SiglipImageProcessor.from_pretrained(
            SIGLIP_ID, subfolder="feature_extractor"
        )
        image_encoder = SiglipVisionModel.from_pretrained(
            SIGLIP_ID, subfolder="image_encoder", torch_dtype=torch.float16
        )
        pipe = HunyuanVideoFramepackPipeline.from_pretrained(
            BACKBONE_ID,
            transformer=transformer,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
        )
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        pipe.enable_model_cpu_offload()
        load_s = round(time.time() - t0, 1)
        vol.commit()  # persist freshly-downloaded weights for warm runs

        img = load_image(image_url)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        call_kwargs = dict(
            image=img,
            prompt=prompt,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=gen,
            sampling_type=sampling_type or "inverted_anti_drifting",
        )
        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt
        t1 = time.time()
        frames = pipe(**call_kwargs).frames[0]
        render_s = round(time.time() - t1, 1)

        out_path = "/tmp/out.mp4"
        export_to_video(frames, out_path, fps=int(fps))
        size = os.path.getsize(out_path)
        s3c.upload_file(
            out_path, bucket, out_key, ExtraArgs={"ContentType": "video/mp4"}
        )

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
                "guidance": guidance, "seed": seed, "fps": fps,
                "sampling_type": call_kwargs["sampling_type"], "i2v": True,
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
    negative_prompt="",
    height=448,
    width=448,
    num_frames=61,
    steps=25,
    guidance=9.0,
    seed=42,
    fps=30,
    out_key="framepack/proof/out.mp4",
    image_url="",
    sampling_type="inverted_anti_drifting",
)


def _merged(kw: dict) -> dict:
    p = dict(_DEFAULTS)
    p.update({k: v for k, v in (kw or {}).items() if k in _DEFAULTS})
    if not p["prompt"]:
        p["prompt"] = "gentle natural motion, cinematic, photoreal detail"
    return p


@app.function(gpu="L40S", memory=65536, **_COMMON)
def render_framepack(**kw):
    return _render(**_merged(kw))


@app.function(secrets=[modal.Secret.from_name("wan-gate")])
@modal.fastapi_endpoint(method="POST")
def kick(body: dict):
    import os

    payload = dict(body or {})
    if payload.pop("k", None) != os.environ.get("WAN_GATE"):
        return {"error": "unauthorized"}
    payload.pop("tier", None)
    call = render_framepack.spawn(**payload)
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
