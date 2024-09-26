from tensorzero import TensorZeroGateway


def generate_haiku(topic):
    return TensorZeroGateway("http://localhost:3000").inference(
        function_name="generate_haiku_with_topic",
        input={"messages": [{"role": "user", "content": {"topic": topic}}]},
    )


print(generate_haiku("artificial intelligence"))
