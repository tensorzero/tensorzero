from tensorzero import TensorZeroGateway


def generate_haiku(topic):
    with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
        return client.inference(
            function_name="generate_haiku_with_topic",
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "arguments": {"topic": topic}}],
                    }
                ],
            },
        )


print(generate_haiku("artificial intelligence"))
