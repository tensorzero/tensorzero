# TensorZero Evals

> [!IMPORTANT]
>
> This is currently work in progress - please bear with us and anticipate breaking changes for now.
>
>

To run the evals binary on an example of fixture data:

- Run TensorZero end-to-end tests - the evals will also run and as part of this do an idempotent insertion of data fixtures into Clickhouse
- Execute `cargo run --features e2e_tests -- --name entity_extraction --variant gpt_4o_mini --config-file ../tensorzero-internal/tests/e2e/tensorzero.toml`

The tests run as part of the top-level unit and end-to-end tests.
