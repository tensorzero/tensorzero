from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_embedded(
    clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero",
    # optional: config_file="path/to/tensorzero.toml",
) as client:
    response = client.inference(
        model_name="openai::gpt-4o-mini",
        # Try other providers easily: "anthropic::claude-3-7-sonnet-20250219"
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Write a haiku about artificial intelligence.",
                }
            ]
        },
    )

print(response)
