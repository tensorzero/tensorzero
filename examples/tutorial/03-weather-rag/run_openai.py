import json

from openai import OpenAI
from tensorzero import TensorZeroGateway

with OpenAI(base_url="http://localhost:3000/openai/v1") as client:
    query_result = client.chat.completions.create(
        model="tensorzero::function_name::generate_weather_query",
        # This is the first inference request in an episode so we don't need to provide an episode_id
        messages=[
            {
                "role": "user",
                "content": "What is the weather like in São Paulo?",
            }
        ],
    )

    print(query_result)

    # In a production setting, you'd validate the output more thoroughly
    assert len(query_result.choices) == 1
    assert query_result.choices[0].message.tool_calls is not None
    assert len(query_result.choices[0].message.tool_calls) == 1
    import json

    tool_call = query_result.choices[0].message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    location = arguments.get("location")
    units = arguments.get("units", "celsius")

    # Now we pretend to make a tool call (e.g. to an API)
    temperature = "35"

    report_result = client.chat.completions.create(
        model="tensorzero::function_name::generate_weather_report",
        # This is the second inference request in an episode so we need to provide the episode_id
        extra_headers={"episode_id": str(query_result.episode_id)},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "location": location,
                        "temperature": temperature,
                        "units": units,
                    }
                ],
            }
        ],
    )

    print(report_result)


with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    feedback_result = client.feedback(
        metric_name="user_rating",
        # Set the episode_id to the one returned in the inference response
        episode_id=report_result.episode_id,
        # Set the value for the metric (numeric types will be coerced to float)
        value=5,
    )

    print(feedback_result)
