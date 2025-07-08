import asyncio

from tensorzero import AsyncTensorZeroGateway, ToolCall


async def main():
    async with await AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000"
    ) as client:
        query_result = await client.inference(
            function_name="generate_weather_query",
            # This is the first inference request in an episode so we don't need to provide an episode_id
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the weather like in São Paulo?",
                    }
                ]
            },
        )

        print(query_result)

        # In a production setting, you'd validate the output more thoroughly
        assert len(query_result.content) == 1
        assert isinstance(query_result.content[0], ToolCall)

        location = query_result.content[0].arguments.get("location")
        units = query_result.content[0].arguments.get("units", "celsius")

        # Now we pretend to make a tool call (e.g. to an API)
        temperature = "35"

        report_result = await client.inference(
            function_name="generate_weather_report",
            # This is the second inference request in an episode so we need to provide the episode_id
            episode_id=query_result.episode_id,
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "arguments": {
                                    "location": location,
                                    "temperature": temperature,
                                    "units": units,
                                },
                            }
                        ],
                    }
                ]
            },
        )

        print(report_result)

        feedback_result = await client.feedback(
            metric_name="user_rating",
            # Set the episode_id to the one returned in the inference response
            episode_id=report_result.episode_id,
            # Set the value for the metric (numeric types will be coerced to float)
            value=5,
        )

        print(feedback_result)


if __name__ == "__main__":
    asyncio.run(main())
