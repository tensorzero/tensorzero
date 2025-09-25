from openai import OpenAI
from tensorzero import patch_openai_client

client = OpenAI()

patch_openai_client(
    client,
    config_file="config/tensorzero.toml",
    async_setup=False,
)

result = client.embeddings.create(
    input="Hello, world!",
    model="tensorzero::embedding_model_name::nomic-embed-text",
)

print(result)
