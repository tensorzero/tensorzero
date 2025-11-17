from tensorzero import TensorZeroGateway

client = TensorZeroGateway.build_http(gateway_url="http://localhost:3000")

messages = [
    "Hi, I'm Sarah Johnson and you can reach me at sarah.j@example.com",
    "My email is contact@company.com",
    "This is John Doe reaching out",
    "I have a question about your product",
]

for message in messages:
    response = client.inference(
        function_name="extract_data",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ]
        },
    )

    print(message)
    print(response)
    print()
