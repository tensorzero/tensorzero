import { z } from "zod";
import type { FeedbackRow } from "~/types/tensorzero";
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
  input_tokens: z.number().nullish(),
  output_tokens: z.number().nullish(),
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

/**
 * Converts a UUIDv7 string to a Date object based on the embedded Unix timestamp.
 * UUIDv7 stores the Unix timestamp (in milliseconds) in its first 48 bits.
 *
 * @param uuid - A string in the canonical UUID format.
 * @returns A Date object corresponding to the UUID's timestamp.
 * @throws Error if the format is invalid or the version nibble is not 7.
 */
export function uuidv7ToTimestamp(uuid: string): Date {
  // Remove all dashes from the UUID
  const hex = uuid.replace(/-/g, "");

  // The canonical UUID should have 32 hex characters after removing dashes
  if (hex.length !== 32) {
    throw new Error("Invalid UUID format");
  }

  // In a canonical UUID, the version nibble is at position 12 (0-indexed).
  if (hex[12] !== "7") {
    throw new Error("Invalid UUID version. Expected version 7.");
  }

  // The first 12 hex digits (48 bits) represent the Unix timestamp in milliseconds.
  const timestampHex = hex.slice(0, 12);
  const timestamp = parseInt(timestampHex, 16);

  return new Date(timestamp);
}
