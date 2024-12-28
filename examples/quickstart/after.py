from tensorzero import TensorZeroGateway

response = TensorZeroGateway("http://localhost:3000").inference(
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
