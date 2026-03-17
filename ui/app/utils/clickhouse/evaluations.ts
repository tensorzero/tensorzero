import type {
  EvaluationResultRow,
  EvaluationRunMetadata,
  RunMetricMetadata,
} from "~/types/tensorzero";

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
  metricsConfig: Record<string, RunMetricMetadata>,
): ConsolidatedEvaluationResult[] {
  const resultMap = new Map<string, ConsolidatedEvaluationResult>();

  for (const result of evaluationResults) {
    // This shouldn't happen in practice, but given the frontend type seemed incorrect, we add this
    // guard to be safe.
    if (!result.metric_name || !result.metric_value) {
      continue;
    }

    const evaluator_name =
      metricsConfig[result.metric_name]?.evaluator_name ?? result.metric_name;

    const metric: ConsolidatedMetric = {
      metric_name: result.metric_name,
      metric_value: result.metric_value,
      evaluator_name,
      evaluator_inference_id: result.evaluator_inference_id ?? undefined,
      is_human_feedback: result.is_human_feedback,
    };

    const key = `${result.datapoint_id}:${result.evaluation_run_id}:${result.variant_name}`;
    const existing = resultMap.get(key);
    if (existing) {
      existing.metrics.push(metric);
    } else {
      const { metric_name, metric_value, ...baseResult } = result;
      resultMap.set(key, {
        ...baseResult,
        metrics: [metric],
      });
    }
  }

  return Array.from(resultMap.values());
}

export interface MergedRunMetrics {
  metrics: RunMetricMetadata[];
  metricsConfig: Record<string, RunMetricMetadata>;
  // Maps full metric_name → short evaluator_name. Keyed by full metric name
  // so that the same evaluator appearing in different contexts (e.g. nested in
  // a named evaluation vs. top-level) gets separate entries.
  evaluatorMetricNames: Record<string, string>;
}

/**
 * Merge metrics from multiple evaluation runs, taking the union keyed by metric name.
 */
export function mergeRunMetrics(
  allMetadata: EvaluationRunMetadata[],
): MergedRunMetrics {
  const mergedMetricsMap = new Map<string, RunMetricMetadata>();
  for (const runMeta of allMetadata) {
    for (const metric of runMeta.metrics) {
      const existing = mergedMetricsMap.get(metric.name);
      if (!existing) {
        mergedMetricsMap.set(metric.name, metric);
      } else {
        // Fill in missing optional fields from later entries
        mergedMetricsMap.set(metric.name, {
          ...metric,
          evaluator_name: existing.evaluator_name ?? metric.evaluator_name,
          optimize: existing.optimize ?? metric.optimize,
        });
      }
    }
  }
  const metrics = [...mergedMetricsMap.values()].sort((a, b) =>
    a.name.localeCompare(b.name),
  );

  const metricsConfig: Record<string, RunMetricMetadata> = {};
  for (const metric of metrics) {
    metricsConfig[metric.name] = metric;
  }

  const evaluatorMetricNames: Record<string, string> = {};
  for (const metric of metrics) {
    if (metric.evaluator_name) {
      evaluatorMetricNames[metric.name] = metric.evaluator_name;
    }
  }

  return { metrics, metricsConfig, evaluatorMetricNames };
}
