# optimizations-server

This is a Python FastAPI based server which implements optimizations recipes used by the UI
Currently supported optimizations:
* OpenAI fine-tuning
* Fireworks fine-tuning

## Usage

The following environment variables are required:
* `TENSORZERO_UI_CONFIG_PATH`: Path to the TensorZero gateway config file used by the ui. Should be the same file used by the NodeJS `ui` server

The following optional environment variables can also be set:
* `OPENAI_BASE_URL`: Overrides the OpenAI server used for fine-tuning jobs
* `FIREWORKS_BASE_URL`: Overrides the Fireworks server used for fine-tuning jobs

To start the server, run `uv run fastapi run src/optimizations_server/main.py --port 7001`

To use the ui fixtures config, run `TENSORZERO_UI_CONFIG_PATH=../ui/fixtures/config/tensorzero.toml uv run fastapi run src/optimizations_server/main.py --port 7001`
