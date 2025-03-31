import { z } from "zod";
import {
  contentBlockOutputSchema,
  jsonInferenceOutputSchema,
  resolvedInputSchema,
} from "./common";

export const EvaluationRunInfoSchema = z.object({
  evaluation_run_id: z.string(),
  variant_name: z.string(),
});

export type EvaluationRunInfo = z.infer<typeof EvaluationRunInfoSchema>;

export const EvaluationResultSchema = z.object({
  datapoint_id: z.string().uuid(),
  evaluation_run_id: z.string().uuid(),
  input: z.string(),
  generated_output: z.string(),
  reference_output: z.string(),
  metric_name: z.string(),
  metric_value: z.string(),
});

export type EvaluationResult = z.infer<typeof EvaluationResultSchema>;

export const EvaluationResultWithVariantSchema = EvaluationResultSchema.extend({
  variant_name: z.string(),
});

export type EvaluationResultWithVariant = z.infer<
  typeof EvaluationResultWithVariantSchema
>;

export const JsonEvaluationResultSchema = z.object({
  datapoint_id: z.string().uuid(),
  evaluation_run_id: z.string().uuid(),
  input: resolvedInputSchema,
  generated_output: jsonInferenceOutputSchema,
  reference_output: jsonInferenceOutputSchema,
  metric_name: z.string(),
  metric_value: z.string(),
});

export type JsonEvaluationResult = z.infer<typeof JsonEvaluationResultSchema>;

export const ChatEvaluationResultSchema = z.object({
  datapoint_id: z.string().uuid(),
  evaluation_run_id: z.string().uuid(),
  input: resolvedInputSchema,
  generated_output: z.array(contentBlockOutputSchema),
  reference_output: z.array(contentBlockOutputSchema),
  metric_name: z.string(),
  metric_value: z.string(),
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
  evalName: string,
  evaluatorName: string,
): string {
  return `tensorzero::evaluation_name::${evalName}::evaluator_name::${evaluatorName}`;
}

function getEvaluatorNameFromMetricName(metricName: string): string {
  const parts = metricName.split("::");
  return parts[parts.length - 1];
}

export const evalInfoResultSchema = z.object({
  evaluation_run_id: z.string().uuid(),
  evaluation_name: z.string(),
  function_name: z.string(),
  variant_name: z.string(),
  last_inference_timestamp: z.string().datetime(),
});

export type EvalInfoResult = z.infer<typeof evalInfoResultSchema>;

export const EvalRunInfoSchema = z.object({
  evaluation_run_id: z.string().uuid(),
  evaluation_name: z.string(),
  function_name: z.string(),
  variant_name: z.string(),
  last_inference_timestamp: z.string().datetime(),
  dataset: z.string(),
});

export type EvalRunInfo = z.infer<typeof EvalRunInfoSchema>;

// Define a type for consolidated metrics
export type ConsolidatedMetric = {
  metric_name: string;
  metric_value: string;
  evaluator_name: string;
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
      });
    }
  }

  // Convert the map values to an array and return
  return Array.from(resultMap.values());
};
