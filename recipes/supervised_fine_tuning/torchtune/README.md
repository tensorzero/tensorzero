# TensorZero Recipe: Supervised Fine-Tuning with Unsloth

The `torchtune.ipynb` notebook provides a step-by-step recipe to perform supervised fine-tuning of models using [torchtune](https://docs.pytorch.org/torchtune/main/) based on data collected by the TensorZero Gateway.

You will need to set a few environment variables in the shell your notebook will run in.
- Set `TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@localhost:8123/tensorzero`.
- Set `HF_TOKEN=<your-hf-token>` to your huggingface token to use models like Llama and Gemma.
- Set `CHECKPOINT_HOME=</path/to/store/large/models>` as the directory to save models downloaded from huggingface.

## Setup

### Optional: Dev Container

We have provided a Dev Container config in `.devcontainer` to help users of VS Code who want to run the notebook on a remote server.
To use our container, follow the [VS Code Instructions](https://code.visualstudio.com/docs/devcontainers/containers#_open-a-folder-on-a-remote-ssh-host-in-a-container), then proceed with the "Using `uv`" instructions below.

### Using [`uv`](https://github.com/astral-sh/uv) (Recommended)

```bash
uv venv  # Create a new virtual environment
source .venv/bin/activate # Activate environment
uv pip sync requirements.txt # Install the dependencies
uv pip install --pre torchtune --extra-index-url https://download.pytorch.org/whl/nightly/cu126
```

### Using `pip`

We recommend using Python 3.10+ and a virtual environment.

```bash
pip install -r requirements.txt
pip install --pre torchtune --extra-index-url https://download.pytorch.org/whl/nightly/cu126
```
