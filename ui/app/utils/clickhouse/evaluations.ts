import { z } from "zod";

import type { EvaluationResultRow } from "~/types/tensorzero";

export const EvaluationRunInfoSchema = z.object({
  evaluation_run_id: z.string(),
  variant_name: z.string(),
  most_recent_inference_date: z.string().datetime(),
});

export type EvaluationRunInfo = z.infer<typeof EvaluationRunInfoSchema>;

export const EvaluationStatisticsSchema = z.object({
  evaluation_run_id: z.string(),
  metric_name: z.string(),
  datapoint_count: z.number(),
  mean_metric: z.number(),
  ci_lower: z.number().nullable(),
  ci_upper: z.number().nullable(),
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
  evaluator_inference_id?: string;
  is_human_feedback: boolean;
};

// Define a type for consolidated evaluation results
export type ConsolidatedEvaluationResult = Omit<
  EvaluationResultRow,
  "metric_name" | "metric_value"
> & {
  metrics: ConsolidatedMetric[];
};

/**
 * Consolidate evaluation results from the API.
 * Groups results by (datapoint_id, evaluation_run_id, variant_name) and collects metrics.
 * Input and output fields are already parsed by the backend.
 */
export function consolidateEvaluationResults(
  evaluationResults: EvaluationResultRow[],
): ConsolidatedEvaluationResult[] {
  // Create a map to store results by datapoint_id and evaluation_run_id
  const resultMap = new Map<string, ConsolidatedEvaluationResult>();

  // Process each evaluation result
  for (const result of evaluationResults) {
    // This shouldn't happen in practice, but given the frontend type seemed incorrect, we add this
    // guard to be safe.
    if (!result.metric_name || !result.metric_value) {
      continue;
    }

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
            evaluator_inference_id: result.evaluator_inference_id ?? undefined,
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
        evaluator_inference_id: result.evaluator_inference_id ?? undefined,
        is_human_feedback: result.is_human_feedback,
      });
    }
  }
  // Convert the map values to an array and return
  return Array.from(resultMap.values());
}
