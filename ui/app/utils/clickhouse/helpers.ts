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
  cost: z.number().nullish(),
});

export type InferenceUsage = z.infer<typeof inferenceUsageSchema>;

export function getTotalInferenceUsage(
  model_inferences: ParsedModelInferenceRow[],
): InferenceUsage {
  return model_inferences.reduce(
    (acc, curr) => {
      // Cost aggregation uses partial-sum semantics: if any model inference
      // reports cost, we sum the known values (treating missing as 0).
      // Only if ALL model inferences lack cost data does the total stay null.
      // This intentionally differs from the Rust backend's poison semantics
      // (where any missing cost makes the total unknown), because in the UI
      // showing partial cost data is more useful than showing nothing.
      const costSum =
        acc.cost == null && curr.cost == null
          ? null
          : (acc.cost ?? 0) + (curr.cost ?? 0);
      return {
        input_tokens: acc.input_tokens + (curr.input_tokens ?? 0),
        output_tokens: acc.output_tokens + (curr.output_tokens ?? 0),
        cost: costSum,
      };
    },
    { input_tokens: 0, output_tokens: 0, cost: null as number | null },
  );
}

/**
 * Formats a cost value (in dollars) for display.
 * Uses adaptive precision based on magnitude to avoid both clutter and
 * misleading rounding. The database stores cost as Decimal(18, 9), so we
 * display up to 8 fractional digits. Values below that floor show "<$0.00000001".
 *
 * Negative costs are possible when caching discounts exceed the base cost.
 * They are displayed with a leading minus sign (e.g. -$0.003).
 *
 * Examples: $0.00001875, $0.003, $1.50, -$0.003, <$0.00000001
 */
export function formatCost(cost: number): string {
  if (!Number.isFinite(cost)) return "$—";
  if (cost === 0) return "$0.00";

  // Handle negative costs: format the absolute value, then prepend "-"
  if (cost < 0) {
    return `-${formatCost(-cost)}`;
  }

  if (cost < 0.00000001) return "<$0.00000001";
  if (cost < 0.001) {
    // For very small costs (< 1/10 cent), show up to 8 decimal places
    return `$${cost.toFixed(8).replace(/\.?0+$/, "")}`;
  }
  if (cost < 0.01) {
    // For small costs (< 1 cent), show up to 6 decimal places
    return `$${cost.toFixed(6).replace(/\.?0+$/, "")}`;
  }
  if (cost < 1) {
    // For costs under $1, show up to 4 decimal places
    return `$${cost.toFixed(4).replace(/\.?0+$/, "")}`;
  }
  // For larger costs, show 2 decimal places
  return `$${cost.toFixed(2)}`;
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
