from openai import OpenAI

oai = OpenAI(api_key="not-used", base_url="http://localhost:3000/openai/v1")

# NB: OpenAI web search can take up to a minute to complete

response = oai.chat.completions.create(
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
