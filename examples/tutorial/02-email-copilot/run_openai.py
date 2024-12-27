from openai import OpenAI
from tensorzero import TensorZeroGateway

with OpenAI(base_url="http://localhost:3000/openai/v1") as client:
    inference_result = client.chat.completions.create(
        model="tensorzero::draft_email",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "recipient_name": "TensorZero Team",
                        "sender_name": "Mark Zuckerberg",
                        "email_purpose": "Acquire TensorZero for $100 billion dollars.",
                    }
                ],
            }
        ],
    )

    print(inference_result)


with TensorZeroGateway("http://localhost:3000") as client:
    feedback_result = client.feedback(
        metric_name="email_draft_accepted",
        # Set the inference_id from the inference response
        inference_id=inference_result.id,
        # Set the value for the metric
        value=True,
    )

    print(feedback_result)
