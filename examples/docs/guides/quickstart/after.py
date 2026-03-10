from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/openai/v1", api_key="not-used")

response = client.chat.completions.create(
    model="tensorzero::function_name::generate_haiku",
    messages=[
        {
            "role": "user",
            "content": "Write a haiku about TensorZero.",
        }
    ],
)

print(response)
