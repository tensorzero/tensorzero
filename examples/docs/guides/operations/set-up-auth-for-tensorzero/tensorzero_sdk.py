import os

from tensorzero import TensorZeroGateway

# Good request

t0 = TensorZeroGateway.build_http(
    api_key=os.environ["TENSORZERO_API_KEY"],  # if not set, SDK automatically reads the environment variable
    gateway_url="http://localhost:3000",
)

response = t0.inference(
    model_name="openai::gpt-5-mini",
    input={
        "messages": [
            {
                "role": "user",
                "content": "Tell me a fun fact.",
            }
        ]
    },
)

print(response)

# Bad request

t0 = TensorZeroGateway.build_http(
    api_key="sk-t0-evilevilevil-hackerhackerhackerhackerhackerhackerhackerhacker",
    gateway_url="http://localhost:3000",
)

try:
    response = t0.inference(
        model_name="openai::gpt-5-mini",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me a fun fact.",
                }
            ]
        },
    )

    print(response)
except Exception as e:
    print(f"Expected error occurred: {e}")
