from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    result = client.inference(
        function_name="extract_email",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "blah blah blah hello@tensorzero.com blah blah blah",
                }
            ]
        },
    )

    print(result)
