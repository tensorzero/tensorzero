from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/openai/v1", api_key="not-used")

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
