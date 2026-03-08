import modal

cuda_version = "12.8.0"
flavor = "devel"
operating_system = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_system}"

sgl_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .pip_install("sglang[all]==0.5.8.post1", "huggingface_hub[hf_transfer]==0.34")
    .apt_install("numactl")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MODELS_DIR = "/models"
MODEL_NAME = "Qwen/Qwen3-1.7B"
MINUTES = 60
SGLANG_PORT = 8000

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
sglang_cache_vol = modal.Volume.from_name("sglang-cache", create_if_missing=True)

N_GPU = 1
app = modal.App(name="sglang-qwen3-inference")


@app.function(
    image=sgl_image,
    gpu=f"L4:{N_GPU}",
    scaledown_window=5 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/sglang": sglang_cache_vol,
    },
    secrets=[modal.Secret.from_name("ro-huggingface-secret")],
    max_containers=1,
)
@modal.concurrent(max_inputs=20)
@modal.web_server(port=SGLANG_PORT, startup_timeout=2 * MINUTES, requires_proxy_auth=True)
def sglang_inference():
    import os
    import subprocess

    os.environ["CUDA_HOME"] = "/usr/local/cuda"

    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        # This prevents the container from OOMing on startup
        "--disable-cuda-graph",
        "--model-path",
        MODEL_NAME,
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen25",
        "--host",
        "0.0.0.0",
        "--port",
        str(SGLANG_PORT),
    ]

    subprocess.Popen(" ".join(cmd), shell=True)
