from tensorzero import TensorZeroGateway

with TensorZeroGateway("http://localhost:3000") as client:
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
    )

print(response)
