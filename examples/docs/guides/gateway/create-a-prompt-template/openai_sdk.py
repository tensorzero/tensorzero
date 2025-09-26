import openai

client = openai.OpenAI(base_url="http://localhost:3000/openai/v1")

result = client.chat.completions.create(
    model="tensorzero::function_name::fun_fact",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "tensorzero::template",  # type: ignore
                    "name": "fun_fact_topic",
                    "arguments": {"topic": "artificial intelligence"},
                }
            ],
        },
    ],
)

print(result)
