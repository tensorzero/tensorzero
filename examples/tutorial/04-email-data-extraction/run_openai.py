from openai import OpenAI

with OpenAI(base_url="http://localhost:3000/openai/v1") as client:
    result = client.chat.completions.create(
        model="tensorzero::function_name::extract_email",
        messages=[
            {
                "role": "user",
                "content": "blah blah blah hello@tensorzero.com blah blah blah",
            },
        ],
    )

    print(result)
