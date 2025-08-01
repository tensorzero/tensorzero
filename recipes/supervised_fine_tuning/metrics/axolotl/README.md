# TensorZero Recipe: Supervised Fine-Tuning with Axolotl

The `axolotl.ipynb` notebook provides a step-by-step recipe to perform supervised fine-tuning of models using [Axolotl](https://axolotl.ai/#learnmore) based on data collected by the TensorZero Gateway.

You will need to set a few environment variables in the shell your notebook will run in.
- Set `TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@localhost:8123/tensorzero`.
- Set `HF_TOKEN=<your-hf-token>` to your huggingface token to use gated models like Llama and Gemma.

## Setup

### Optional: Dev Container

We have provided a Dev Container config in `.devcontainer` to help users of VS Code who want to run the notebook on a remote server.
To use our container, follow the [VS Code Instructions](https://code.visualstudio.com/docs/devcontainers/containers#_open-a-folder-on-a-remote-ssh-host-in-a-container), then proceed with the "Using `uv`" instructions below.

### Using [`uv`](https://github.com/astral-sh/uv) (Recommended)

```bash
export UV_TORCH_BACKEND=cu126
uv venv  # Create a new virtual environment
source .venv/bin/activate # Activate environment
uv pip sync requirements.txt # Install the dependencies
uv pip install --no-build-isolation axolotl[flash-attn,deepspeed]
```

### Using `pip`

We recommend using Python 3.10+ and a virtual environment.

```bash
pip install -r requirements.txt
pip install --no-build-isolation axolotl[flash-attn,deepspeed]
```
