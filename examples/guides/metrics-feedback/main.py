from openai import OpenAI
from tensorzero import TensorZeroGateway

# Use the OpenAI SDK for inference
client = OpenAI(base_url="http://localhost:3000/openai/v1", api_key="not-used")

inference_response = client.chat.completions.create(
    model="tensorzero::function_name::generate_haiku",
    messages=[
        {
            "role": "user",
            "content": "Write a haiku about TensorZero.",
        }
    ],
)

print(inference_response)

# Use the TensorZero SDK for feedback
t0 = TensorZeroGateway.build_http(gateway_url="http://localhost:3000")

feedback_response = t0.feedback(
    metric_name="haiku_rating",
    inference_id=inference_response.id,  # alternatively, you can assign feedback to an episode_id
    value=True,  # let's assume it deserves a 👍
)

print(feedback_response)

demonstration_response = t0.feedback(
    metric_name="demonstration",
    inference_id=inference_response.id,
    value="Silicon dreams float\nMinds born of human design\nLearning without end",  # the haiku we wish the LLM had written
)

print(demonstration_response)

comment_response = t0.feedback(
    metric_name="comment",
    inference_id=inference_response.id,
    value="Never mention you're an artificial intelligence, AI, bot, or anything like that.",
)

print(comment_response)
