from openai import OpenAI  # or AsyncOpenAI
from tensorzero import patch_openai_client

client = OpenAI()
patch_openai_client(client, async_setup=False)

response = client.chat.completions.create(
    model="tensorzero::model_name::anthropic::claude-3-5-haiku-20241022",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of Japan?",
        }
    ],
)

print(response)
