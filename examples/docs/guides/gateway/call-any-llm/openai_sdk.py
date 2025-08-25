from openai import OpenAI
from tensorzero import patch_openai_client

client = OpenAI()
patch_openai_client(client, async_setup=False)

response = client.chat.completions.create(
    model="tensorzero::model_name::openai::gpt-5-mini",
    # or: model="tensorzero::model_name::anthropic::claude-sonnet-4-20250514"
    # or: Google, AWS, Azure, xAI, vLLM, Ollama, and many more
    messages=[
        {
            "role": "user",
            "content": "Tell me a fun fact.",
        }
    ],
)

print(response)
