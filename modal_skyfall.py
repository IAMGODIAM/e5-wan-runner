"""E5 Sovereign LLM Lane — Skyfall-31B-v4.2 (creative/uncensored) on Modal via vLLM.

Serves an OpenAI-compatible /v1 API (chat/completions, completions, models) so
guaardvark's OpenAI-compatible provider (GUAARDVARK_MISTRAL_BASE_URL) can point
straight at it. Arch = MistralForCausalLM, ~62.7GB bf16, fits one A100-80GB with
room for a 16k KV cache. Auth = Bearer <WAN_GATE> (vLLM --api-key), reusing the
wan-gate secret so no new credential is introduced.

Deployed web endpoint (base URL for guaardvark):
  https://iamgodiam--e5-skyfall-llm-serve.modal.run/v1
Served model name: "skyfall-31b". Scale-to-zero after idle; first call cold-loads
the weights from the shared HF volume.
"""
import modal

MODEL_ID = "TheDrummer/Skyfall-31B-v4.2"
SERVED_NAME = "skyfall-31b"
VLLM_PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm",
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_DO_NOT_TRACK": "1",
        }
    )
)
app = modal.App("e5-skyfall-llm", image=image)
vol = modal.Volume.from_name("e5-wan-hf", create_if_missing=True)


@app.function(
    gpu="A100-80GB",
    volumes={"/vol/hf": vol},
    secrets=[modal.Secret.from_name("wan-gate")],
    timeout=3600,
    scaledown_window=300,
    max_containers=1,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=1800)
def serve():
    import os, subprocess

    key = os.environ["WAN_GATE"]
    cmd = [
        "vllm", "serve", MODEL_ID,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--api-key", key,
        "--served-model-name", SERVED_NAME,
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.92",
        "--download-dir", "/vol/hf",
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
