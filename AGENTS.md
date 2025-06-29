# Python Dependencies

We use `uv` to manage Python dependencies.

When updating Python dependencies anywhere in the project, you must update both the `uv.lock` and `requirements.txt` to keep them in sync.

1. Update `pyproject.toml` with your changes
2. Run `uv lock --project="pyproject.toml"` from the directory containing the `pyproject.toml` to generate/update `uv.lock`
3. Run `uv export --project="pyproject.toml" --output-file="requirements.txt"` from the same directory to generate/update `requirements.txt`

The pre-commit hooks automatically handle this by running `uv lock` and `uv export` for all `pyproject.toml` files in the repository.
