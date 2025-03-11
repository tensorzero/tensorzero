import openai


def generate_haiku(topic):
    with openai.OpenAI(base_url="http://localhost:3000/openai/v1") as client:
        return client.chat.completions.create(
            model="tensorzero::function_name::generate_haiku_with_topic",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "tensorzero::arguments": {"topic": topic}}
                    ],
                },
            ],
        )


print(generate_haiku("artificial intelligence"))
