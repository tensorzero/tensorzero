from openai import OpenAI
from tensorzero import patch_openai_client

client = OpenAI()

patch_openai_client(
    client,
    clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero",
    # optional: config_file="path/to/tensorzero.toml",
    async_setup=False,
)

response = client.chat.completions.create(
    model="tensorzero::model_name::openai::gpt-4o-mini",
    # Try other providers easily: "tensorzero::model_name::anthropic::claude-sonnet-4-5-20250929",
    messages=[
        {
            "role": "user",
            "content": "Write a haiku about TensorZero.",
        }
    ],
)

print(response)
