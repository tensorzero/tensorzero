import { FeedbackRequestSchema, TensorZeroClient } from "./tensorzero";

if (!process.env.TENSORZERO_GATEWAY_URL) {
  throw new Error("TENSORZERO_GATEWAY_URL environment variable is required");
}

// Export a singleton instance
export const tensorZeroClient = new TensorZeroClient(
  process.env.TENSORZERO_GATEWAY_URL,
);

export async function addHumanFeedback(formData: FormData) {
  const metricName = formData.get("metricName");
  const value = formData.get("value");
  const episodeId = formData.get("episodeId");
  const inferenceId = formData.get("inferenceId");
  const tags = {
    "tensorzero::human_feedback": "true",
  };
  if ((episodeId && inferenceId) || (!episodeId && !inferenceId)) {
    throw new Error(
      "Exactly one of episodeId and inferenceId should be provided",
    );
  }
  console.log("foo");
  const feedbackRequest = FeedbackRequestSchema.parse({
    metric_name: metricName,
    value,
    episode_id: episodeId,
    inference_id: inferenceId,
    tags,
  });
  const response = await tensorZeroClient.feedback(feedbackRequest);
  return response;
}
