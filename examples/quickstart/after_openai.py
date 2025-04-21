import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

with OpenAI(
        base_url="http://localhost:3000/openai/v1",
        api_key=os.environ["OPENAI_API_KEY"]
    ) as client:
    response = client.chat.completions.create(
        model="tensorzero::function_name::generate_haiku",
        messages=[
            {
                "role": "user",
                "content": "Write a haiku about artificial intelligence.",
            }
        ],
    )

print(response)
