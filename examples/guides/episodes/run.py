from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    haiku_response = client.inference(
        function_name="generate_haiku",
        # We don't provide an episode_id for the first inference in the episode
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Write a haiku about artificial intelligence.",
                }
            ]
        },
    )

    print(haiku_response)

    # When we don't provide an episode_id, the gateway will generate a new one for us
    episode_id = haiku_response.episode_id

    # In a production application, we'd first validate the response to ensure the model returned the correct fields
    haiku = haiku_response.content[0].text

    analysis_response = client.inference(
        function_name="analyze_haiku",
        # For future inferences in that episode, we provide the episode_id that we received
        episode_id=episode_id,
        input={
            "messages": [
                {
                    "role": "user",
                    "content": f"Write a one-paragraph analysis of the following haiku:\n\n{haiku}",
                }
            ]
        },
    )

    print(analysis_response)
