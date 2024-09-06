import asyncio
import os
from pprint import pprint

from tensorzero import AsyncTensorZeroGateway


async def main(gateway_url: str):
    async with AsyncTensorZeroGateway(gateway_url) as client:
        inference_result = await client.inference(
            function_name="draft_email",
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "recipient_name": "TensorZero Team",
                            "sender_name": "Mark Zuckerberg",
                            "email_purpose": "Acquire TensorZero for $100 billion dollars.",
                        },
                    }
                ]
            },
        )

        # If everything is working correctly, the `variant_name` field should change depending on the request
        pprint(inference_result)

        feedback_result = await client.feedback(
            metric_name="email_draft_accepted",
            # Set the inference_id from the inference response
            inference_id=inference_result.inference_id,
            # Set the value for the metric
            value=True,
        )

        pprint(feedback_result)

        print("Success! 🎉")


if __name__ == "__main__":
    gateway_url = os.getenv("TENSORZERO_GATEWAY_URL")
    if not gateway_url:
        gateway_url = "http://localhost:3000"

    asyncio.run(main(gateway_url))
