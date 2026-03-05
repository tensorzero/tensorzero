from openai import OpenAI

with OpenAI(base_url="http://localhost:3000/openai/v1", api_key="not-used") as client:
    haiku_response = client.chat.completions.create(
        model="tensorzero::function_name::generate_haiku",
        # We don't provide an episode_id for the first inference in the episode
        messages=[
            {
                "role": "user",
                "content": "Write a haiku about TensorZero.",
            }
        ],
    )

    print(haiku_response)

    # When we don't provide an episode_id, the gateway will generate a new one for us
    episode_id = haiku_response.episode_id

    # In a production application, we'd first validate the response to ensure the model returned the correct fields
    haiku = haiku_response.choices[0].message.content

    analysis_response = client.chat.completions.create(
        model="tensorzero::function_name::analyze_haiku",
        # For future inferences in that episode, we provide the episode_id that we received
        extra_body={"tensorzero::episode_id": str(episode_id)},
        messages=[
            {
                "role": "user",
                "content": f"Write a one-paragraph analysis of the following haiku:\n\n{haiku}",
            }
        ],
    )

    print(analysis_response)
