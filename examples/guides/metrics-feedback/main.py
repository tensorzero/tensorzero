from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    inference_response = client.inference(
        function_name="generate_haiku",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Write a haiku about TensorZero.",
                }
            ]
        },
    )

    print(inference_response)

    feedback_response = client.feedback(
        metric_name="haiku_rating",
        inference_id=inference_response.inference_id,  # alternatively, you can assign feedback to an episode_id
        value=True,  # let's assume it deserves a 👍
    )

    print(feedback_response)

    demonstration_response = client.feedback(
        metric_name="demonstration",
        inference_id=inference_response.inference_id,
        value="Silicon dreams float\nMinds born of human design\nLearning without end",  # the haiku we wish the LLM had written
    )

    print(demonstration_response)

    comment_response = client.feedback(
        metric_name="comment",
        inference_id=inference_response.inference_id,
        value="Never mention you're an artificial intelligence, AI, bot, or anything like that.",
    )

    print(comment_response)
