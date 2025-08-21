import tensorzero
from openai import OpenAI
from tensorzero import patch_openai_client

print(tensorzero.**version**)

client = OpenAI()

patch_openai_client(client, async_setup=False)

result = client.embeddings.create(
input="Hello, world!",
model="text-embedding-3-small",
)

print(result)
