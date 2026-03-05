from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/openai/v1", api_key="not-used")

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
