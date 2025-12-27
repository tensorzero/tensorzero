from openai import OpenAI

with OpenAI() as client:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Write a haiku about TensorZero.",
            }
        ],
    )

print(response)
