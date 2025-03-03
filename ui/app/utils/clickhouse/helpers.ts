import { z } from "zod";
import type { FeedbackRow } from "./feedback";
import type { ParsedModelInferenceRow } from "./inference";

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

export const inferenceUsageSchema = z.object({
  input_tokens: z.number(),
  output_tokens: z.number(),
});

export type InferenceUsage = z.infer<typeof inferenceUsageSchema>;

export function getTotalInferenceUsage(
  model_inferences: ParsedModelInferenceRow[],
): InferenceUsage {
  return model_inferences.reduce(
    (acc, curr) => {
      return {
        input_tokens: acc.input_tokens + (curr.input_tokens ?? 0),
        output_tokens: acc.output_tokens + (curr.output_tokens ?? 0),
      };
    },
    { input_tokens: 0, output_tokens: 0 },
  );
}
