import { getConfig } from "./config/index.server";
import {
  FeedbackRequestSchema,
  TensorZeroClient,
  type FeedbackResponse,
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
  const tags: Record<string, string> = {
    "tensorzero::human_feedback": "true",
  };
  const datapointId = formData.get("datapointId");
  if (datapointId) {
    tags["tensorzero::datapoint_id"] = datapointId.toString();
  }
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

type AddEvaluationHumanFeedbackResponse = {
  feedbackResponse: FeedbackResponse;
  judgeDemonstrationResponse: FeedbackResponse | null;
};

export async function addEvaluationHumanFeedback(
  formData: FormData,
): Promise<AddEvaluationHumanFeedbackResponse> {
  const [r1, r2] = await Promise.all([
    addHumanFeedback(formData),
    addJudgeDemonstration(formData),
  ]);
  // We don't need the feedback ID for the judge demonstration as long as it succeeds
  return {
    feedbackResponse: r1,
    judgeDemonstrationResponse: r2,
  };
}

export async function addJudgeDemonstration(formData: FormData) {
  console.log("formData", formData);
  const evaluatorInferenceId = formData.get("evaluatorInferenceId")?.toString();
  if (!evaluatorInferenceId) {
    // This is likely not an LLM Judge datapoint since this ID is not present.
    return null;
  }
  const value = formData.get("value")?.toString();
  if (!value) {
    throw new Error("Value is required");
  }
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
  let parsedValue: JSONValue;
  if (metricType === "float") {
    parsedValue = parseFloat(value);
  } else if (metricType === "boolean") {
    parsedValue = value === "true";
  } else {
    throw new Error(`Unsupported metric type: ${metricType}`);
  }
  const demonstrationValue = { score: parsedValue };
  const feedbackRequest = FeedbackRequestSchema.parse({
    metric_name: "demonstration",
    value: demonstrationValue,
    episode_id: null,
    inference_id: evaluatorInferenceId,
    tags: { "tensorzero::human_feedback": "true" },
    internal: true,
  });
  const response = await tensorZeroClient.feedback(feedbackRequest);
  return response;
}
