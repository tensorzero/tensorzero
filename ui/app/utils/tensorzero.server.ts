import { getConfig } from "./config/index.server";
import {
  FeedbackRequestSchema,
  TensorZeroClient,
  type JSONValue,
} from "./tensorzero";

if (!process.env.TENSORZERO_GATEWAY_URL) {
  throw new Error("TENSORZERO_GATEWAY_URL environment variable is required");
}

// Export a singleton instance
export const tensorZeroClient = new TensorZeroClient(
  process.env.TENSORZERO_GATEWAY_URL,
);

export async function addHumanFeedback(formData: FormData) {
  const metricName = formData.get("metricName")?.toString();
  if (!metricName) {
    throw new Error("Metric name is required");
  }
  const config = await getConfig();
  const metric = config.metrics[metricName];
  if (!metric) {
    throw new Error(`Metric ${metricName} not found`);
  }
  const metricType = metric.type;
  // Metrics can be of type boolean, float, comment, or demonstration.
  // In this case we need to handle the value differently depending on the metric type.
  const formValue = formData.get("value")?.toString();
  if (!formValue) {
    throw new Error("Value is required");
  }
  let value: JSONValue;
  if (metricType === "boolean") {
    value = formValue === "true";
  } else if (metricType === "float") {
    value = parseFloat(formValue);
  } else if (metricType === "comment") {
    value = formValue;
  } else if (metricType === "demonstration") {
    value = JSON.parse(formValue);
  } else {
    throw new Error(`Unsupported metric type: ${metricType}`);
  }
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
  const feedbackRequest = FeedbackRequestSchema.parse({
    metric_name: metricName,
    value,
    episode_id: episodeId,
    inference_id: inferenceId,
    tags,
    internal: true,
  });
  const response = await tensorZeroClient.feedback(feedbackRequest);
  return response;
}
