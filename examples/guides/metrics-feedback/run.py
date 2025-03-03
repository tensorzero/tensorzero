from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    inference_response = client.inference(
        function_name="generate_haiku",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Write a haiku about artificial intelligence.",
                }
            ]
        },
    )

    print(inference_response)

    feedback_response = client.feedback(
        metric_name="haiku_rating",
        inference_id=inference_response.inference_id,
        value=True,  # let's assume it deserves a 👍
    )

    print(feedback_response)
