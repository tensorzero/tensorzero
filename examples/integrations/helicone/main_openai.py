import os

from openai import OpenAI

# Retrieve the Helicone API key from the environment variable
helicone_api_key = os.getenv("HELICONE_API_KEY")
assert helicone_api_key, "HELICONE_API_KEY is not set"


# Build our OpenAI client
client = OpenAI(
    base_url="http://localhost:3000/openai/v1",  # our local TensorZero Gateway
)


# Test our `helicone_gpt_4o_mini` model
response = client.chat.completions.create(
    model="tensorzero::model_name::helicone_gpt_4o_mini",
    messages=[
        {"role": "user", "content": "Who is the CEO of OpenAI?"},
    ],
    extra_body={
        "tensorzero::extra_headers": [
            {
                "model_name": "helicone_gpt_4o_mini",
                "provider_name": "helicone",
                "name": "Helicone-Auth",
                "value": f"Bearer {helicone_api_key}",
            },
        ]
    },
)

print(response)


# Test our `helicone_grok_3` model
response = client.chat.completions.create(
    model="tensorzero::model_name::helicone_grok_3",
    messages=[
        {"role": "user", "content": "Who is the CEO of xAI?"},
    ],
    extra_body={
        "tensorzero::extra_headers": [
            {
                "model_name": "helicone_grok_3",
                "provider_name": "helicone",
                "name": "Helicone-Auth",
                "value": f"Bearer {helicone_api_key}",
            },
        ]
    },
)

print(response)
