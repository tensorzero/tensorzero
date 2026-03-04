from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/openai/v1", api_key="not-used")

# NB: OpenAI web search can take up to a minute to complete

response = client.chat.completions.create(
    # The model is defined in config/tensorzero.toml
    model="tensorzero::model_name::gpt-5-mini-responses-web-search",
    messages=[
        {
            "role": "user",
            "content": "What is the current population of Japan?",
        }
    ],
)

print(response)
