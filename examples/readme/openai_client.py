from openai import OpenAI
from tensorzero import patch_openai_client

client = patch_openai_client(
    OpenAI(),
    clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero",
    config_file="config/tensorzero.toml",
    async_setup=False,
)

response = client.chat.completions.create(
    model="tensorzero::model_name::openai::gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Write a haiku about artificial intelligence.",
        }
    ],
)

print(response)
