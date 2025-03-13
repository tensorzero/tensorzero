# TensorZero Evals

::tip:: This is currently work in progress -- please bear with us and anticipate breaking changes for now.

To run the evals binary on fixture data:
* run TensorZero end-to-end tests -- the evals will also run and as part of this do an idempotent insertion of data fixtures into Clickhouse
* execute `cargo run --name eval_name -v variant_name --config_file ../tensorzero-internal/tests/e2e/tensorzero.toml`

The tests run as part of the top-level unit and end-to-end tests.
