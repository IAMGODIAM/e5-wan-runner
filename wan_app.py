"""E5 Sovereign Video Lane — Wan 2.2 TI2V-5B proof render on Beam GPU (A10G).

Walking-skeleton slice for the Wan2GP / Wan 2.2 adoption verdict: proves the
headless text-to-video path end-to-end (HF model -> Beam GPU -> MP4 -> R2)
with measured load/render wall times. All generation params ride the enqueue
payload, so future runs (720p, more frames, i2v later) need no redeploy.
"""
from beam import task_queue, Image, Volume

image = (
    Image(python_version="python3.11")
    .add_commands([
        "apt-get update -y",
        "apt-get install -y ffmpeg",
    ])
    .add_python_packages([
        "torch",
        "diffusers>=0.35.0",
        "transformers>=4.46.0",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "ftfy",
        "imageio",
        "imageio-ffmpeg",
        "opencv-python-headless",
        "boto3",
        "huggingface_hub[hf_transfer]",
    ])
)

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


@task_queue(
    name="e5-wan-video",
    gpu="A10G",
    cpu=8.0,
    memory="32Gi",
    image=image,
    timeout=3600,
    volumes=[Volume(name="e5-hf-cache", mount_path="/vol/hf")],
    secrets=["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ACCOUNT_ID", "R2_BUCKET"],
)
def generate(
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
            MODEL_ID, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        load_s = round(time.time() - t0, 1)

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
                "guidance": guidance, "seed": seed, "fps": fps, "model": MODEL_ID,
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
