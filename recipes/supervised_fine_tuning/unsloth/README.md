# TensorZero Recipe: Supervised Fine-Tuning with Unsloth

The `unsloth.ipynb` notebook provides a step-by-step recipe to perform supervised fine-tuning of models using [Unsloth](https://unsloth.ai) based on data collected by the TensorZero Gateway.
Set `TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@localhost:8123/tensorzero` in the shell your notebook will run in.

## Setup

### Optional: Dev Container

We have provided a Dev Container config in `.devcontainer` to help users of VS Code who want to run the notebook on a remote server.
The Dev Container pulls the [Unsloth docker image](https://hub.docker.com/r/unsloth/unsloth).
To use our container, follow the [VS Code Instructions](https://code.visualstudio.com/docs/devcontainers/containers#_open-a-folder-on-a-remote-ssh-host-in-a-container), then proceed with the "Using `uv`" instructions below.

### Using [`uv`](https://github.com/astral-sh/uv) (Recommended)

```bash
uv venv  # Create a new virtual environment
source .venv/bin/activate # Activate environment
uv pip install -r requirements.txt # Install the dependencies
```
