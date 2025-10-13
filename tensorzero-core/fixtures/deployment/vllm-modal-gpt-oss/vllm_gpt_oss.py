# Based on https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/gpt_oss_inference.py
# ---
# pytest: false
# ---

# # Run OpenAI's gpt-oss model with vLLM

# ## Background

# [gpt-oss](https://openai.com/index/introducing-gpt-oss/) is a reasoning model
# that comes in two flavors: `gpt-oss-120B` and `gpt-oss-20B`. They are both Mixture
# of Experts (MoE) models with a low number of active parameters, ensuring they
# combine good world knowledge and capabilities with fast inference.

# We describe a few of its notable features below.

# ### MXFP4

# OpenAI's gpt-oss models use a fairly uncommon 4bit [`mxfp4`](https://arxiv.org/abs/2310.10537) floating point
# format for the MoE layers. This "block" quantization format combines `e2m1` floating point numbers
# with blockwise scaling factors. The attention operations are not quantized.

# ### Attention Sinks

# Attention sink models allow for longer context lengths without sacrificing output quality. The vLLM team
# added [attention sink support](https://huggingface.co/kernels-community/vllm-flash-attn3)
# for Flash Attention 3 (FA3) in preparation for this release.

# ### Response Format

# GPT-OSS is trained with the [harmony response format](https://github.com/openai/harmony) which enables models
# to output to multiple channels for chain-of-thought (CoT) and input tool-calling preambles along with regular text responses.
# We'll stick to a simpler format here, but see [this cookbook](https://cookbook.openai.com/articles/openai-harmony)
# for details on the new format.

# ## Set up the container image

# We'll start by defining a [custom container `Image`](https://modal.com/docs/guide/custom-container) that
# installs all the necessary dependencies to run vLLM and the model. This includes a special-purpose vLLM prerelease
# and a nightly PyTorch install for Triton support.


import modal

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.10.1+gptoss",
        "huggingface_hub[hf_transfer]==0.34",
        pre=True,
        extra_options="--extra-index-url https://wheels.vllm.ai/gpt-oss/ --extra-index-url https://download.pytorch.org/whl/nightly/cu128 --index-strategy unsafe-best-match",
    )
)


# ## Download the model weights

# We'll be downloading OpenAI's model from Hugging Face. We're running
# the 20B parameter model by default but you can easily switch to [the 120B model](https://huggingface.co/openai/gpt-oss-120b),
# which also fits in a single H100 or H200 GPU.

MODEL_NAME = "openai/gpt-oss-20b"
MODEL_REVISION = "f47b95650b3ce7836072fb6457b362a795993484"

# Although vLLM will download weights from Hugging Face on-demand, we want to
# cache them so we don't do it every time our server starts. We'll use [Modal Volumes](https://modal.com/docs/guide/volumes)
# for our cache. Modal Volumes are essentially a "shared disk" that all Modal
# Functions can access like it's a regular disk. For more on storing model
# weights on Modal, see [this guide](https://modal.com/docs/guide/model-weights).

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# The first time you run a new model or configuration with vLLM on a fresh machine,
# a number of artifacts are created. We also cache these artifacts.

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# There are a number of compilation settings for vLLM. Compilation improves inference performance
# but incur extra latency at engine start time. We offer a high-level variable for controlling this trade-off.

FAST_BOOT = False  # slower boots but faster inference

# Among the artifacts that are created at startup are CUDA graphs,
# which allow the replay of several kernel launches for the price of one,
# reducing CPU overhead. We over-ride the defaults with a smaller number of sizes
# that we think better balances latency from future JIT CUDA graph generation
# and startup latency.

MAX_INPUTS = 32  # how many requests can one replica handle? tune carefully!
CUDA_GRAPH_CAPTURE_SIZES = [  # 1, 2, 4, ... MAX_INPUTS
    1 << i for i in range((MAX_INPUTS).bit_length())
]

# ## Build a vLLM engine and serve it

# The function below spawns a vLLM instance listening at port 8000, serving requests to our model.

app = modal.App("vllm-gpt-oss-20b")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=5 * MINUTES,  # how long should we stay up with no requests?
    timeout=5 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES, requires_proxy_auth=True)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    if not FAST_BOOT:  # CUDA graph capture is only used with `--enforce-eager`
        cmd += ["-O.cudagraph_capture_sizes=" + str(CUDA_GRAPH_CAPTURE_SIZES).replace(" ", "")]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)
