from tensorzero import TensorZeroGateway

t0 = TensorZeroGateway.build_http(gateway_url="http://localhost:3000")

for i in range(5):
    response = t0.inference(
        model_name="openai::gpt-4.1-mini",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me a fun fact.",
                }
            ]
        },
        params={
            "chat_completion": {
                "max_tokens": 1000,
            }
        },
        tags={
            # "x_id": "1",
            "x_id": f"{i}",
            # "x_id": "tensorzero::all"
        },
    )

    print(response)
