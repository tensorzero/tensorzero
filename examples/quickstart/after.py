from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_embedded(
    postgres_url="postgres://postgres:postgres@localhost:5432/tensorzero",
    config_file="config/tensorzero.toml",
) as client:
    response = client.inference(
        function_name="generate_haiku",
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
