from tensorzero import TensorZeroGateway

t0 = TensorZeroGateway.build_http(gateway_url="http://localhost:3000")

response = t0.inference(
    model_name="openai::gpt-5-mini",
    # or: model="anthropic::claude-sonnet-4-20250514"
    # or: Google, AWS, Azure, xAI, vLLM, Ollama, and many more
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
