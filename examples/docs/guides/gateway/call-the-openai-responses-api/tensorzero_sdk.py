from tensorzero import TensorZeroGateway

t0 = TensorZeroGateway.build_http(gateway_url="http://localhost:3000")

# NB: OpenAI web search can take up to a minute to complete

response = t0.inference(
    # The model is defined in config/tensorzero.toml
    # Thought summaries are enabled in the config via extra_body
    model_name="gpt-5-mini-responses-web-search",
    input={
        "messages": [
            {
                "role": "user",
                "content": "What is the current population of Japan?",
            }
        ]
    },
)

print(response)
