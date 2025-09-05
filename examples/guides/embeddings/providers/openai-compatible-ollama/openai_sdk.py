from openai import OpenAI
from tensorzero import patch_openai_client

client = OpenAI(api_key="none")

patch_openai_client(
    client,
    config_file="config/tensorzero.toml",
    clickhouse_url=None,
    async_setup=False,
)

result = client.embeddings.create(
    input="Hello, world!",
    model="tensorzero::embedding_model_name::nomic-embed-text",
)

print(result)
