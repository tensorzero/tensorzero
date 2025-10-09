# Rust

- Use `cargo check` for quick verification, restrict further (e.g. `cargo check --package tensorzero-core`) if appropriate. Test suite compilation is slow.
- Run unit tests with `cargo test-unit` which uses `nextest` under the hood.
- Run `cargo clippy --all-targets --all-features -- -D warnings` once you're done with your work (before committing) to catch warnings and errors.
- If you update Rust types or functions used in TypeScript, regenerate bindings with `pnpm build-bindings` from `internal/tensorzero-node` (not root). Run `cargo check` first to catch compilation errors.
- If you change the signature of a struct, function, and so on, use `rg` to find all instances in the codebase. For example, search for `StructName {` when updating struct fields.

# Python Dependencies

We use `uv` to manage Python dependencies.

When updating Python dependencies anywhere in the project, you must update both the `uv.lock` and `requirements.txt` to keep them in sync.

1. Update `pyproject.toml` with your changes
2. Run `uv lock --project="pyproject.toml"` from the directory containing the `pyproject.toml` to generate/update `uv.lock`
3. Run `uv export --project="pyproject.toml" --output-file="requirements.txt"` from the same directory to generate/update `requirements.txt`

The pre-commit hooks automatically handle this by running `uv lock` and `uv export` for all `pyproject.toml` files in the repository.

# CI/CD

- Most GitHub Actions workflows run on Unix only, but some also run on Windows and macOS. For workflows that run on multiple operating systems, ensure any bash scripts are compatible with all three platforms. You can check which OS a workflow uses by looking at the `runs-on` field. Setting `shell: bash` in the job definition is often sufficient.

# UI

- After modifying UI code, run from the `ui/` directory: `pnpm run format`, `pnpm run lint`, `pnpm run typecheck`. All commands must pass.

# Misc

- `CONTRIBUTING.md` has additional context on working on this codebase.
- `rg` should be available by default. Install it if it's missing.
