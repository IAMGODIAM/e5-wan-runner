"""E5 Sovereign Video Lane — LTX-2.3 on Modal. The sound-native cinema tier.

One tier: H100 80GB running Lightricks' official DistilledPipeline
(ltx-2.3-22b-distilled-1.1, bf16 checkpoint + fp8-cast quantization — dodges the
pre-#172 fp8-file dequant bug), Gemma-3-12B QAT encoder from Lightricks' own
ungated mirror. Video + native stereo audio muxed into one MP4 by the official
encode_video. Same HTTP contract as the wan lane: POST kick (gate) -> {call_id};
GET stat -> {done,result}. Renders write MP4 + timing JSON to R2.

License note: LTX-2 Community License — worldwide grant, free under $10M
aggregated annual revenue (E5 qualifies); AUP requires disclosing machine-
generated content where published.
"""
import modal

LTX_COMMIT = "4f8905737aac86a554637cac86c178877a39c744"  # main @ 2026-08-03, post-#172
CKPT_REPO = "Lightricks/LTX-2.3"
CKPT_FILE = "ltx-2.3-22b-distilled-1.1.safetensors"
UPSAMPLER_FILE = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_REPO = "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .run_commands(
        "git clone https://github.com/Lightricks/LTX-2.git /opt/ltx2 "
        f"&& cd /opt/ltx2 && git checkout {LTX_COMMIT}",
        "pip install --no-cache-dir /opt/ltx2/packages/ltx-core /opt/ltx2/packages/ltx-pipelines",
    )
    .uv_pip_install(
        "boto3",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]",
        "requests",
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)
app = modal.App("e5-ltx-video-modal", image=image)
vol = modal.Volume.from_name("e5-ltx-hf", create_if_missing=True)

_COMMON = dict(
    timeout=3600,
    volumes={"/vol/hf": vol},
    secrets=[modal.Secret.from_name("r2-creds")],
)

_DEFAULTS = dict(
    prompt="",
    negative_prompt="",  # accepted, ignored: distilled runs CFG=1
    height=704,
    width=1280,
    num_frames=121,
    steps=8,             # informational; distilled sigmas are fixed
    guidance=1.0,        # informational
    seed=42,
    fps=24,
    out_key="ltx/proof/out.mp4",
    image_url="",
    image_strength=1.0,
)


def _snap_frames(f):
    f = max(9, min(481, int(f)))
    return ((f - 1) // 8) * 8 + 1


def _snap_dim(v):
    v = max(256, min(1920, int(v)))
    return (v // 64) * 64


def _merged(kw):
    p = dict(_DEFAULTS)
    p.update({k: v for k, v in (kw or {}).items() if k in _DEFAULTS})
    if not p["prompt"]:
        p["prompt"] = (
            "A majestic black dragon with gold-trimmed scales soars over a sunlit Miami "
            "skyline at golden hour, slow cinematic camera orbit, volumetric light, "
            "photoreal detail. Deep resonant wingbeats, wind rushing past, a low "
            "orchestral swell building with brass."
        )
    return p


def _render(prompt, negative_prompt, height, width, num_frames, steps, guidance,
            seed, fps, out_key, image_url, image_strength):
    import json, os, time, traceback

    os.environ.setdefault("HF_HOME", "/vol/hf")

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
        t0 = time.time()
        from huggingface_hub import hf_hub_download, snapshot_download

        ckpt = hf_hub_download(CKPT_REPO, CKPT_FILE, cache_dir="/vol/hf")
        upsampler = hf_hub_download(CKPT_REPO, UPSAMPLER_FILE, cache_dir="/vol/hf")
        gemma_root = snapshot_download(GEMMA_REPO, cache_dir="/vol/hf")
        vol.commit()  # persist weights for warm runs

        import torch
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.args import ImageConditioningInput
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

        try:
            from ltx_core.quantization import QuantizationKind
            quant = QuantizationKind("fp8-cast").to_policy(checkpoint_path=ckpt)
        except Exception:
            quant = None  # fall back to bf16-resident (H100 80GB holds it)

        height = _snap_dim(height)
        width = _snap_dim(width)
        num_frames = _snap_frames(num_frames)

        images = []
        if image_url:
            import requests as rq
            r = rq.get(image_url, timeout=60)
            r.raise_for_status()
            ext = ".png" if "png" in (r.headers.get("content-type") or "") else ".jpg"
            src = f"/tmp/cond{ext}"
            with open(src, "wb") as f:
                f.write(r.content)
            images.append(ImageConditioningInput(path=src, frame_idx=0, strength=float(image_strength)))

        pipe = DistilledPipeline(
            distilled_checkpoint_path=ckpt,
            gemma_root=gemma_root,
            spatial_upsampler_path=upsampler,
            loras=[],
            quantization=quant,
        )
        load_s = round(time.time() - t0, 1)

        t1 = time.time()
        tiling = TilingConfig.default()
        chunks = get_video_chunks_number(num_frames, tiling)
        with torch.inference_mode():
            video, audio = pipe(
                prompt=prompt,
                seed=int(seed),
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=float(fps),
                images=images,
                tiling_config=tiling,
            )
            out_path = "/tmp/out.mp4"
            encode_video(video=video, fps=float(fps), audio=audio,
                         output_path=out_path, video_chunks_number=chunks)
        render_s = round(time.time() - t1, 1)

        size = os.path.getsize(out_path)
        s3c.upload_file(out_path, bucket, out_key, ExtraArgs={"ContentType": "video/mp4"})
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        result = {
            "rc": 0, "out_key": out_key, "bytes": size, "gpu": gpu,
            "load_s": load_s, "render_s": render_s, "audio": True,
            "params": {"h": height, "w": width, "frames": num_frames, "seed": seed,
                        "fps": fps, "model": f"{CKPT_REPO}/{CKPT_FILE}",
                        "i2v": bool(image_url), "quant": "fp8-cast" if quant else "bf16"},
        }
        s3c.put_object(Bucket=bucket, Key=out_key + ".json",
                       Body=json.dumps(result, indent=2).encode(), ContentType="application/json")
        print(json.dumps(result))
        return result
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        try:
            s3c.put_object(Bucket=bucket, Key=err_key, Body=tb.encode())
        except Exception as e2:
            print("could not upload error.txt:", e2)
        return {"error": str(e), "error_key": err_key}


@app.function(gpu="H100", memory=131072, cpu=16.0, **_COMMON)
def render_ltx(**kw):
    return _render(**_merged(kw))


@app.function(secrets=[modal.Secret.from_name("wan-gate")])
@modal.fastapi_endpoint(method="POST")
def kick(body: dict):
    import os
    payload = dict(body or {})
    if payload.pop("k", None) != os.environ.get("WAN_GATE"):
        return {"error": "unauthorized"}
    payload.pop("tier", None)
    call = render_ltx.spawn(**payload)
    return {"call_id": call.object_id, "tier": "ltx"}


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
