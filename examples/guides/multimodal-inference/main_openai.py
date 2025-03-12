from openai import OpenAI

with OpenAI(base_url="http://localhost:3000/openai/v1") as client:
    response = client.chat.completions.create(
        model="tensorzero::model_name::openai::gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Do the images share any common features?",
                    },
                    # Remote image of Ferris the crab
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
                        },
                    },
                    # One-pixel orange image encoded as a base64 string
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdj+O/P8B8ABe0CTsv8mHgAAAAASUVORK5CYII=",
                        },
                    },
                ],
            }
        ],
    )

    print(response)
