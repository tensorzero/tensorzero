from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3000/openai/v1",
    api_key="unused",
)

messages_list = [
    "Hi, I'm Sarah Johnson and you can reach me at sarah.j@example.com",
    "My email is contact@company.com",
    "This is John Doe reaching out",
    "I have a question about your product",
]

for message in messages_list:
    response = client.chat.completions.create(
        model="tensorzero::function_name::extract_data",
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
    )

    print(message)
    print(response)
    print()
