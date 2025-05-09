from tensorzero import TensorZeroGateway  # or AsyncTensorZeroGateway

UNSTRUCTURED_TEXT = """
Tech giant TensorZero announced on Tuesday the acquisition of AI startup Anthropic for $1.2 trillion.
The deal, expected to close by the end of Q3 2027, will help TensorZero expand its machine learning capabilities according to CEO Gabriel Bianconi.
"This strategic acquisition positions us to better serve the open source community," Bianconi stated during a press conference in New York.
Anthropic, founded in 2021 by Dario Amodei and Daniela Amodei, reported $85 billion in revenue last year and currently employs over 2000 people globally.
""".strip()


with TensorZeroGateway.build_http(
    gateway_url="http://localhost:3000",
) as t0:
    # Standard inference
    response = t0.inference(
        function_name="extract_names",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": UNSTRUCTURED_TEXT,
                }
            ]
        },
    )

    print(response)

    print()

    # Streaming inference
    for chunk in t0.inference(
        function_name="extract_names",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": UNSTRUCTURED_TEXT,
                }
            ]
        },
        stream=True,
    ):
        print(chunk)

    print()

    # Inference with a dynamic output schema
    response = t0.inference(
        function_name="extract_names",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": UNSTRUCTURED_TEXT,
                }
            ]
        },
        output_schema={
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
    )

    print(response)
