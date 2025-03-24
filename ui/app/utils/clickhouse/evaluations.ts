import { z } from "zod";

export const EvaluationRunInfoSchema = z.object({
  eval_run_id: z.string(),
  variant_name: z.string(),
});

export type EvaluationRunInfo = z.infer<typeof EvaluationRunInfoSchema>;

export const EvaluationResultSchema = z.object({
  datapoint_id: z.string().uuid(),
  eval_run_id: z.string().uuid(),
  input: z.string(),
  generated_output: z.string(),
  reference_output: z.string(),
  metric_name: z.string(),
  metric_value: z.string(),
});

export type EvaluationResult = z.infer<typeof EvaluationResultSchema>;

export const EvaluationStatisticsSchema = z.object({
  eval_run_id: z.string(),
  metric_name: z.string(),
  datapoint_count: z.number(),
  mean_metric: z.number(),
  stderr_metric: z.number(),
});

export type EvaluationStatistics = z.infer<typeof EvaluationStatisticsSchema>;
