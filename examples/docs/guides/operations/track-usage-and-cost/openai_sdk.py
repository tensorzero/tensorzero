from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/openai/v1")

response = client.chat.completions.create(
    model="tensorzero::model_name::gpt-5",  # see config/tensorzero.toml
    messages=[
        {
            "role": "user",
            "content": "Write a haiku about TensorZero.",
        }
    ],
)

print(response)
