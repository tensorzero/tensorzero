import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.9.1", "huggingface_hub[hf_transfer]==0.32.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MODELS_DIR = "/models"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MINUTES = 60
VLLM_PORT = 8000

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

N_GPU = 1
app = modal.App(name="vllm-inference-qwen")


@app.function(
    image=vllm_image,
    gpu=f"T4:{N_GPU}",
    scaledown_window=5 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("vllm-secret")],
    max_containers=1,
)
@modal.concurrent(max_inputs=20)  #  how many concurrent requests can one container handle
@modal.web_server(port=VLLM_PORT, startup_timeout=2 * MINUTES, requires_proxy_auth=True)
def vllm_inference():
    import os
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        os.environ["VLLM_API_KEY"],
        "--tool-call-parser",
        "hermes",
        "--enable-auto-tool-choice",
        "--dtype",
        "half",
    ]

    subprocess.Popen(" ".join(cmd), shell=True)
