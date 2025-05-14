from openai import OpenAI

with OpenAI(base_url="http://localhost:3000/openai/v1") as client:
    response = client.chat.completions.create(
        model="tensorzero::function_name::generate_haiku",
        messages=[
            {
                "role": "user",
                "content": "Write a haiku about artificial intelligence.",
            }
        ],
    )

print(response)
