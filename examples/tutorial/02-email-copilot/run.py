from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    inference_result = client.inference(
        function_name="draft_email",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "arguments": {
                                "recipient_name": "TensorZero Team",
                                "sender_name": "Mark Zuckerberg",
                                "email_purpose": "Acquire TensorZero for $100 billion dollars.",
                            },
                        }
                    ],
                }
            ],
        },
    )

    # If everything is working correctly, the `variant_name` field should change depending on the request
    print(inference_result)

    feedback_result = client.feedback(
        metric_name="email_draft_accepted",
        # Set the inference_id from the inference response
        inference_id=inference_result.inference_id,
        # Set the value for the metric
        value=True,
    )

    print(feedback_result)
