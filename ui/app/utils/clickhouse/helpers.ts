import { z } from "zod";
import type { FeedbackRow } from "./feedback";

// Since demonstrations and comments do not have a metric_name, we need to
// infer the metric name from the structure of the feedback row
export const getMetricName = (feedback: FeedbackRow) => {
  if ("metric_name" in feedback) {
    return feedback.metric_name;
  }
  if ("inference_id" in feedback) {
    return "demonstration";
  }
  return "comment";
};

export const parseFeedbackData = <T>(
  rawData: unknown,
  schema: z.ZodType<T>,
): T => {
  const result = schema.safeParse(rawData);
  if (!result.success) {
    throw new Error("Invalid data format");
  }
  return result.data;
};
