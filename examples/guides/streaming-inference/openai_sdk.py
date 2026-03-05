import os

from openai import OpenAI


def main(gateway_url: str):
    with OpenAI(base_url=f"{gateway_url}/openai/v1", api_key="not-used") as client:
        stream = client.chat.completions.create(
            model="tensorzero::function_name::chatbot",
            messages=[
                {
                    "role": "user",
                    "content": "Share an extensive list of fun facts about Japan.",
                },
            ],
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="")


if __name__ == "__main__":
    gateway_url = os.getenv("TENSORZERO_GATEWAY_URL")
    if not gateway_url:
        gateway_url = "http://localhost:3000"

    main(gateway_url)
