# TensorZero Recipe: Supervised Fine-Tuning with Fireworks

The `fireworks.ipynb` notebook provides a step-by-step recipe to perform supervised fine-tuning with Fireworks based on data collected by the TensorZero Gateway.

## Setup

1. Create a `.env` file with the `FIREWORKS_API_KEY` environment variable (see `.env.example` for an example).
2. Configure your gateway config with `[provider_types.fireworks.sft]` section containing your `account_id`:
   ```toml
   [provider_types.fireworks.sft]
   account_id = "your-fireworks-account-id"
   ```
3. Run `docker compose up` to launch the TensorZero Gateway, the TensorZero UI, and a development ClickHouse database (run the [quickstart guide](https://www.tensorzero.com/docs/quickstart/) or an example in /examples if your ClickHouse database is not yet populated with data).
4. Run the `fireworks.ipynb` Jupyter notebook.

### Using [`uv`](https://github.com/astral-sh/uv) (Recommended)

```bash
uv venv  # Create a new virtual environment
uv pip sync requirements.txt  # Install the dependencies
```

### Using `pip`

We recommend using Python 3.10+ and a virtual environment.

```bash
pip install -r requirements.txt
```
