from openai import OpenAI

oai = OpenAI(base_url="http://localhost:3000/openai/v1")


def call_llm(user_id):
    try:
        return oai.chat.completions.create(
            model="tensorzero::model_name::openai::gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Tell me a fun fact.",
                }
            ],
            max_tokens=1000,
            extra_body={"tensorzero::tags": {"user_id": user_id}},
        )
    except Exception as e:
        print(f"Error calling LLM: {e}")


# The second should fail
print(call_llm("intern"))
print(call_llm("intern"))  # should return None

# Both should work
print(call_llm("ceo"))
print(call_llm("ceo"))
