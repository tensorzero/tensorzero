from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/openai/v1", api_key="not-used")

result = client.embeddings.create(
    input="Hello, world!",
    model="tensorzero::embedding_model_name::text-embedding-3-small",
)

print(result)
