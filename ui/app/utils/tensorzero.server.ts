import { getConfig } from "./config/index.server";
import {
  FeedbackRequestSchema,
  TensorZeroClient,
  TensorZeroServerError,
  type FeedbackResponse,
  type JSONValue,
} from "~/utils/tensorzero";

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
    throw new TensorZeroServerError.InvalidMetricName(
      "Metric name is required",
    );
  }
  const config = await getConfig();
  const metric = config.metrics[metricName];
  if (!metric) {
    throw new TensorZeroServerError.UnknownMetric(
      `Metric ${metricName} not found`,
    );
  }
  const metricType = metric.type;
  // Metrics can be of type boolean, float, comment, or demonstration.
  // In this case we need to handle the value differently depending on the metric type.
  const formValue = formData.get("value");
  if (!formValue || typeof formValue !== "string") {
    throw new TensorZeroServerError.InputValidation("Value is required");
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
    throw new TensorZeroServerError.InputValidation(
      `Unsupported metric type: ${metricType}`,
    );
  }
  const episodeId = formData.get("episodeId");
  const inferenceId = formData.get("inferenceId");
  const tags: Record<string, string> = {
    "tensorzero::human_feedback": "true",
  };
  // We should either get both of these or none of them
  const datapointId = formData.get("datapointId");
  const evaluatorInferenceId = formData.get("evaluatorInferenceId");
  if (datapointId && evaluatorInferenceId) {
    tags["tensorzero::datapoint_id"] = datapointId.toString();
    tags["tensorzero::evaluator_inference_id"] =
      evaluatorInferenceId.toString();
  } else if (!datapointId && !evaluatorInferenceId) {
    // Do nothing
  } else {
    throw new TensorZeroServerError.InputValidation(
      "Either both or neither of datapointId and evaluatorInferenceId should be provided",
    );
  }
  if ((episodeId && inferenceId) || (!episodeId && !inferenceId)) {
    throw new TensorZeroServerError.InputValidation(
      "Exactly one of episodeId and inferenceId should be provided",
    );
  }
  const feedbackRequest = FeedbackRequestSchema.safeParse({
    metric_name: metricName,
    value,
    episode_id: episodeId,
    inference_id: inferenceId,
    tags,
    internal: true,
  });
  if (!feedbackRequest.success) {
    throw new TensorZeroServerError.InputValidation(
      "Feedback request schema validation failed",
    );
  }
  const response = await tensorZeroClient.feedback(feedbackRequest.data);
  return response;
}

type AddEvaluationHumanFeedbackResponse = {
  feedbackResponse: FeedbackResponse;
  judgeDemonstrationResponse: FeedbackResponse | null;
};

export async function addEvaluationHumanFeedback(
  formData: FormData,
): Promise<AddEvaluationHumanFeedbackResponse> {
  // We check that the formData contains a datapointId
  const datapointId = formData.get("datapointId")?.toString();
  if (!datapointId) {
    throw new Error(
      "Datapoint ID is required. This is a bug. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports",
    );
  }
  const [r1, r2] = await Promise.all([
    addHumanFeedback(formData),
    addJudgeDemonstration(formData),
  ]);
  return {
    feedbackResponse: r1,
    judgeDemonstrationResponse: r2,
  };
}

export async function addJudgeDemonstration(formData: FormData) {
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
