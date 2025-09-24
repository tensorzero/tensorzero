from tensorzero import TensorZeroGateway

t0 = TensorZeroGateway.build_http(gateway_url="http://localhost:3000")

result = t0.inference(
    function_name="fun_fact",
    input={
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "template",
                        "name": "fun_fact_topic",
                        "arguments": {"topic": "artificial intelligence"},
                    }
                ],
            }
        ],
    },
)

print(result)
