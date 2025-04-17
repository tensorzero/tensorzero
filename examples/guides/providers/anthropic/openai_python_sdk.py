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


# By the way, you can...
#
# - Run `patch_openai_client` asynchronously: `await patch_openai_client(client)`
# - Set up custom configuration and observability: `patch_openai_client(config_file="...", clickhouse_url="...", ...)`
# - Use a standalone HTTP TensorZero Gateway: use `OpenAI(base_url="...")` instead of `patch_openai_client`
# - Call custom models and functions: `model_name="my_model"` or `function_name="my_function"`
