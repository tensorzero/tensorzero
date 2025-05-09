from openai import OpenAI

messages = [
    {
        "role": "system",
        "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
    },
    {"role": "user", "content": "how can I solve 8x + 7 = -23"},
]

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "math_response",
        "schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "explanation": {"type": "string"},
                            "output": {"type": "string"},
                        },
                        "required": ["explanation", "output"],
                        "additionalProperties": False,
                    },
                },
                "final_answer": {"type": "string"},
            },
            "required": ["steps", "final_answer"],
            "additionalProperties": False,
            "strict": True,
        },
    },
}

openai = OpenAI()

response = openai.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=messages,
    response_format=response_format,
)

print(response)
print()

openait0 = OpenAI(base_url="http://localhost:3000/openai/v1")

response2 = openait0.chat.completions.create(
    model="tensorzero::model_name::openai::gpt-4o-2024-08-06",
    messages=messages,
    response_format=response_format,
)

print(response2)
