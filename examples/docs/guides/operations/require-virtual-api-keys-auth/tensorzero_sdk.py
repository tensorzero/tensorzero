from tensorzero import TensorZeroGateway

t0 = TensorZeroGateway.build_http(
    # api_key="xxx",  # if not set, SDK reads from TENSORZERO_API_KEY env var
    gateway_url="http://localhost:3000",
)

response = t0.inference(
    model_name="openai::gpt-5-mini",
    input={
        "messages": [
            {
                "role": "user",
                "content": "Tell me a fun fact.",
            }
        ]
    },
)

print(response)
