import os

from tensorzero import TensorZeroGateway

# Retrieve the Helicone API key from the environment variable
helicone_api_key = os.getenv("HELICONE_API_KEY")
assert helicone_api_key, "HELICONE_API_KEY is not set"

# Build our TensorZero gateway
with TensorZeroGateway.build_http(
    gateway_url="http://localhost:3000",
) as t0:
    # Test our `helicone_gpt_4o_mini` model
    response = t0.inference(
        model_name="helicone_gpt_4o_mini",
        input={
            "messages": [
                {"role": "user", "content": "Who is the CEO of OpenAI?"},
            ],
        },
        extra_headers=[
            {
                "model_name": "helicone_gpt_4o_mini",
                "provider_name": "helicone",
                "name": "Helicone-Auth",
                "value": f"Bearer {helicone_api_key}",
            },
        ],
    )

    print(response)

    # Test our `helicone_grok_3` model
    response = t0.inference(
        model_name="helicone_grok_3",
        input={
            "messages": [
                {"role": "user", "content": "Who is the CEO of xAI?"},
            ],
        },
        extra_headers=[
            {
                "model_name": "helicone_grok_3",
                "provider_name": "helicone",
                "name": "Helicone-Auth",
                "value": f"Bearer {helicone_api_key}",
            },
        ],
    )

    print(response)
