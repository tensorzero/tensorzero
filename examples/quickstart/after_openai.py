from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/openai/v1")

response = client.chat.completions.create(
    model="tensorzero::generate_haiku",
    messages=[
        {
            "role": "user",
            "content": "Write a haiku about artificial intelligence.",
        }
    ],
)

print(response)
