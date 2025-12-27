from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_embedded(
    clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero",
    # optional: config_file="path/to/tensorzero.toml",
) as client:
    response = client.inference(
        model_name="openai::gpt-4o-mini",
        # Try other providers easily: "anthropic::claude-sonnet-4-5-20250929"
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Write a haiku about TensorZero.",
                }
            ]
        },
    )

print(response)
