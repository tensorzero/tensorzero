from tensorzero import TensorZeroGateway

t0 = TensorZeroGateway.build_http(gateway_url="http://localhost:3000")


def call_llm(user_id):
    return t0.inference(
        model_name="openai::gpt-4.1-mini",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me a fun fact.",
                }
            ]
        },
        # We have rate limits on tokens, so we must be conservative and provider `max_tokens`
        params={
            "chat_completion": {
                "max_tokens": 1000,
            }
        },
        tags={
            "user_id": user_id,
        },
    )


# the second should fail
print(call_llm("intern"))
print(call_llm("intern"))
print(call_llm("intern"))

# should all work
print(call_llm("ceo"))
print(call_llm("ceo"))
