from openai import OpenAI

with OpenAI() as client:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Write a haiku about artificial intelligence.",
                    },
                    {
                        "type": "text",
                        "text": "In german.",
                    },
                ],
            }
        ],
    )

print(response)
