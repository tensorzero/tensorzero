from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(
    gateway_url="http://localhost:3000",
) as client:
    response = client.inference(
        model_name="openai::gpt-4o-mini",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Do the images share any common features?",
                        },
                        # Remote image of Ferris the crab
                        {
                            "type": "image",
                            "url": "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
                        },
                        # One-pixel orange image encoded as a base64 string
                        {
                            "type": "image",
                            "mime_type": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdj+O/P8B8ABe0CTsv8mHgAAAAASUVORK5CYII=",
                        },
                    ],
                }
            ],
        },
    )

    print(response)
