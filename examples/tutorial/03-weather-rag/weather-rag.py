import asyncio
import os
from pprint import pprint

from tensorzero import AsyncTensorZeroGateway, ToolCall


async def main(gateway_url: str):
    async with AsyncTensorZeroGateway(gateway_url) as client:
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

        pprint(query_result)

        # In a production setting, you'd validate the output more thoroughly
        assert len(query_result.content) == 1
        assert isinstance(query_result.content[0], ToolCall)

        location = query_result.content[0].arguments.get("location")
        units = query_result.content[0].arguments.get("units")
        temperature = "80"  # imagine this came from some API

        report_result = await client.inference(
            function_name="generate_weather_report",
            # This is the second inference request in an episode so we need to provide the episode_id
            episode_id=query_result.episode_id,
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "location": location,
                            "temperature": temperature,
                            "units": units,
                        },
                    }
                ]
            },
        )

        pprint(report_result)

        feedback_result = await client.feedback(
            metric_name="user_rating",
            # Set the episode_id to the one returned in the inference response
            episode_id=report_result.episode_id,
            # Set the value for the metric (numeric types will be coerced to float)
            value=5,
        )

        pprint(feedback_result)

        print("Success! 🎉")


if __name__ == "__main__":
    gateway_url = os.getenv("TENSORZERO_GATEWAY_URL")
    if not gateway_url:
        gateway_url = "http://localhost:3000"

    asyncio.run(main(gateway_url))
