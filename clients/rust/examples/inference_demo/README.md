# Inference demo

This demo shows performing inference with both an HTTP gateway server and an embedded gateway

## Usage

1. Run `docker compose up` in `<tensorzero_repository>/examples/haiku-hidden-preferences`

The following steps should be run from the root of the repository

2. To perform inference against the running gateway server, run:

```bash
cargo run --example inference_demo -- --gateway-url http://localhost:3000 --function-name 'judge_haiku' --streaming '{"topic": "Rivers", "haiku": "Endless roaring flow. Mountains weep streams for oceans. Carve earth like giants"}'
```

3. To perform inference against a embedded gateway server (running within the example binary), run:

```bash
CLICKHOUSE_URL=http://127.0.0.1:8123/tensorzero cargo run --example inference_demo -- --config-path examples/haiku-hidden-preferences/config/tensorzero.toml --function-name judge_haiku --streaming '{"topic": "Rivers", "haiku": "Endless roaring flow. Mountains weep streams for oceans. Carve earth like giants"}'
```

The '--streaming' flag controls whether or not the output is streamed to the console as it becomes available, or only disabled when the full response is available.
