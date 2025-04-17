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


# By the way, you can...
#
# - Set up custom configuration and observability: `build_embedded(config_file="...", clickhouse_url="...")`
# - Use a standalone HTTP TensorZero Gateway: use `build_http` instead of `build_embedded`
# - Call custom models and functions: `model_name="my_model"` or `function_name="my_function"`
