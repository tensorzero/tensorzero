import { z } from "zod";
import {
  contentBlockChatOutputSchema,
  jsonInferenceOutputSchema,
  displayInputSchema,
} from "./common";

export const EvaluationRunInfoSchema = z.object({
  evaluation_run_id: z.string(),
  variant_name: z.string(),
  most_recent_inference_date: z.string().datetime(),
});

export type EvaluationRunInfo = z.infer<typeof EvaluationRunInfoSchema>;

export const EvaluationRunSearchResultSchema = z.object({
  evaluation_run_id: z.string(),
  variant_name: z.string(),
});

export type EvaluationRunSearchResult = z.infer<
  typeof EvaluationRunSearchResultSchema
>;

export const EvaluationResultSchema = z.object({
  datapoint_id: z.string().uuid(),
  evaluation_run_id: z.string().uuid(),
  input: z.string(),
  generated_output: z.string(),
  reference_output: z.string(),
  dataset_name: z.string(),
  metric_name: z.string(),
  metric_value: z.string(),
  is_human_feedback: z.boolean(),
});

export type EvaluationResult = z.infer<typeof EvaluationResultSchema>;

export const EvaluationResultWithVariantSchema = EvaluationResultSchema.extend({
  variant_name: z.string(),
});

export type EvaluationResultWithVariant = z.infer<
  typeof EvaluationResultWithVariantSchema
>;

export const JsonEvaluationResultSchema = z.object({
  inference_id: z.string().uuid(),
  episode_id: z.string().uuid(),
  datapoint_id: z.string().uuid(),
  evaluation_run_id: z.string().uuid(),
  evaluator_inference_id: z.string().uuid().nullable(),
  input: displayInputSchema,
  generated_output: jsonInferenceOutputSchema,
  reference_output: jsonInferenceOutputSchema.nullable(),
  dataset_name: z.string(),
  metric_name: z.string(),
  metric_value: z.string(),
  feedback_id: z.string().uuid(),
  is_human_feedback: z.boolean(),
  name: z.string().nullable(),
  staled_at: z.string().datetime().nullable(),
});

export type JsonEvaluationResult = z.infer<typeof JsonEvaluationResultSchema>;

export const ChatEvaluationResultSchema = z.object({
  inference_id: z.string().uuid(),
  episode_id: z.string().uuid(),
  datapoint_id: z.string().uuid(),
  evaluation_run_id: z.string().uuid(),
  evaluator_inference_id: z.string().uuid().nullable(),
  input: displayInputSchema,
  generated_output: z.array(contentBlockChatOutputSchema),
  reference_output: z.array(contentBlockChatOutputSchema).nullable(),
  dataset_name: z.string(),
  metric_name: z.string(),
  metric_value: z.string(),
  feedback_id: z.string().uuid(),
  is_human_feedback: z.preprocess((val) => val === 1, z.boolean()),
  name: z.string().nullable(),
  staled_at: z.string().datetime().nullable(),
});

export type ChatEvaluationResult = z.infer<typeof ChatEvaluationResultSchema>;

export const ParsedEvaluationResultSchema = z.union([
  JsonEvaluationResultSchema,
  ChatEvaluationResultSchema,
]);

export type ParsedEvaluationResult = z.infer<
  typeof ParsedEvaluationResultSchema
>;

export const JsonEvaluationResultWithVariantSchema =
  JsonEvaluationResultSchema.extend({
    variant_name: z.string(),
  });

export type JsonEvaluationResultWithVariant = z.infer<
  typeof JsonEvaluationResultWithVariantSchema
>;

export const ChatEvaluationResultWithVariantSchema =
  ChatEvaluationResultSchema.extend({
    variant_name: z.string(),
  });

export type ChatEvaluationResultWithVariant = z.infer<
  typeof ChatEvaluationResultWithVariantSchema
>;

export const ParsedEvaluationResultWithVariantSchema = z.union([
  JsonEvaluationResultWithVariantSchema,
  ChatEvaluationResultWithVariantSchema,
]);

export type ParsedEvaluationResultWithVariant = z.infer<
  typeof ParsedEvaluationResultWithVariantSchema
>;

export const EvaluationStatisticsSchema = z.object({
  evaluation_run_id: z.string(),
  metric_name: z.string(),
  datapoint_count: z.number(),
  mean_metric: z.number(),
  stderr_metric: z.number().nullable(),
});

export type EvaluationStatistics = z.infer<typeof EvaluationStatisticsSchema>;

export function getEvaluatorMetricName(
  evaluationName: string,
  evaluatorName: string,
): string {
  return `tensorzero::evaluation_name::${evaluationName}::evaluator_name::${evaluatorName}`;
}

function getEvaluatorNameFromMetricName(metricName: string): string {
  const parts = metricName.split("::");
  return parts[parts.length - 1];
}

export const evaluationInfoResultSchema = z.object({
  evaluation_run_id: z.string().uuid(),
  evaluation_name: z.string(),
  dataset_name: z.string(),
  function_name: z.string(),
  variant_name: z.string(),
  last_inference_timestamp: z.string().datetime(),
});

export type EvaluationInfoResult = z.infer<typeof evaluationInfoResultSchema>;

// Define a type for consolidated metrics
export type ConsolidatedMetric = {
  metric_name: string;
  metric_value: string;
  evaluator_name: string;
  evaluator_inference_id: string | null;
  is_human_feedback: boolean;
};

// Define a type for consolidated evaluation results
export type ConsolidatedEvaluationResult = Omit<
  ParsedEvaluationResultWithVariant,
  "metric_name" | "metric_value"
> & {
  metrics: ConsolidatedMetric[];
};

export const consolidate_evaluation_results = (
  evaluation_results: ParsedEvaluationResultWithVariant[],
): ConsolidatedEvaluationResult[] => {
  // Create a map to store results by datapoint_id and evaluation_run_id
  const resultMap = new Map<string, ConsolidatedEvaluationResult>();

  // Process each evaluation result
  for (const result of evaluation_results) {
    const key = `${result.datapoint_id}:${result.evaluation_run_id}:${result.variant_name}`;

    if (!resultMap.has(key)) {
      // Create a new consolidated result without metric_name and metric_value
      const { metric_name, metric_value, ...baseResult } = result;

      resultMap.set(key, {
        ...baseResult,
        metrics: [
          {
            metric_name,
            metric_value,
            evaluator_name: getEvaluatorNameFromMetricName(metric_name),
            evaluator_inference_id: result.evaluator_inference_id,
            is_human_feedback: result.is_human_feedback,
          },
        ],
      });
    } else {
      // Add this metric to the existing result
      const existingResult = resultMap.get(key)!;
      existingResult.metrics.push({
        metric_name: result.metric_name,
        metric_value: result.metric_value,
        evaluator_name: getEvaluatorNameFromMetricName(result.metric_name),
        evaluator_inference_id: result.evaluator_inference_id,
        is_human_feedback: result.is_human_feedback,
      });
    }
  }

  // Convert the map values to an array and return
  return Array.from(resultMap.values());
};
