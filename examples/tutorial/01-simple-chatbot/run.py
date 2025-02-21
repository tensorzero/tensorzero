from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    result = client.inference(
        function_name="mischievous_chatbot",
        input={
            "system": "You are a friendly but mischievous AI assistant.",
            "messages": [
                {"role": "user", "content": "What is the capital of Japan?"},
            ],
        },
    )

print(result)
