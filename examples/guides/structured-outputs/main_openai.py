from openai import OpenAI  # or AsyncOpenAI

UNSTRUCTURED_TEXT = """
Tech giant TensorZero announced on Tuesday the acquisition of AI startup Anthropic for $1.2 trillion.
The deal, expected to close by the end of Q3 2027, will help TensorZero expand its machine learning capabilities according to CEO Gabriel Bianconi.
"This strategic acquisition positions us to better serve the open source community," Bianconi stated during a press conference in New York.
Anthropic, founded in 2021 by Dario Amodei and Daniela Amodei, reported $85 billion in revenue last year and currently employs over 2000 people globally.
""".strip()


client = OpenAI(base_url="http://localhost:3000/openai/v1")

# Standard
response = client.chat.completions.create(
    model="tensorzero::function_name::extract_names",
    messages=[
        {
            "role": "user",
            "content": UNSTRUCTURED_TEXT,
        }
    ],
)

print(response)
print()

# Streaming
stream = client.chat.completions.create(
    model="tensorzero::function_name::extract_names",
    messages=[
        {
            "role": "user",
            "content": UNSTRUCTURED_TEXT,
        }
    ],
    stream=True,
)

for chunk in stream:
    print(chunk)


# Inference with a dynamic output schema
response = client.chat.completions.create(
    model="tensorzero::function_name::extract_names",
    messages=[
        {
            "role": "user",
            "content": UNSTRUCTURED_TEXT,
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "people": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "title": {"type": "string"},
                        },
                        "required": ["name", "title"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["people"],
            "additionalProperties": False,
        },
    },
)

print(response)
