from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_embedded(
    clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero",
    config_file="config/tensorzero.toml",
) as client:
    response = client.inference(
        function_name="generate_haiku",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Write a haiku about artificial intelligence.",
                }
            ]
        },
        output_schema={
            "type": "object",
            "properties": {"haiku": {"type": "string"}, "rating": {"type": "integer"}},
            "required": ["haiku", "rating"],
            "additionalProperties": False,
        },
        params={
            "chat_completion": {
                "json_mode": "strict",
            }
        },
    )

print(response)
