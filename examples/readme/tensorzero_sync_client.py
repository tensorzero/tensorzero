from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    response = client.inference(
        model_name="openai::gpt-4o-mini",
        # Try other providers easily: "anthropic::claude-3-7-sonnet-20240229",
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
