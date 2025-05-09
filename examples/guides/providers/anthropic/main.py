from tensorzero import TensorZeroGateway  # or AsyncTensorZeroGateway

with TensorZeroGateway.build_embedded() as t0:
    response = t0.inference(
        model_name="anthropic::claude-3-5-haiku-20241022",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of Japan?",
                }
            ]
        },
    )

    print(response)
