import os

from openai import OpenAI

# Good request

client = OpenAI(
    api_key=os.environ["TENSORZERO_API_KEY"],
    base_url="http://localhost:3000/openai/v1",
)

response = client.chat.completions.create(
    model="tensorzero::model_name::openai::gpt-5-mini",
    messages=[
        {
            "role": "user",
            "content": "Tell me a fun fact.",
        }
    ],
)

print(response)

# Bad request

bad_client = OpenAI(
    api_key="sk-t0-evilevilevil-hackerhackerhackerhackerhackerhackerhackerhacker",
    base_url="http://localhost:3000/openai/v1",
)

try:
    response = bad_client.chat.completions.create(
        model="tensorzero::model_name::openai::gpt-5-mini",
        messages=[
            {
                "role": "user",
                "content": "Tell me a fun fact.",
            }
        ],
    )

    print(response)
except Exception as e:
    print(f"Expected error occurred: {e}")
