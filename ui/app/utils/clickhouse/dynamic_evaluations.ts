import { z } from "zod";

export const dynamicEvaluationRunSchema = z
  .object({
    name: z.string().nullable(),
    id: z.string().uuid(),
    variant_pins: z.record(z.string(), z.string()),
    tags: z.record(z.string(), z.string()),
    project_name: z.string().nullable(),
    timestamp: z.string().datetime(),
  })
  .strict();

export type DynamicEvaluationRun = z.infer<typeof dynamicEvaluationRunSchema>;

export const dynamicEvaluationRunWithEpisodeCountSchema =
  dynamicEvaluationRunSchema.extend({
    num_episodes: z.number().default(0),
  });

export type DynamicEvaluationRunWithEpisodeCount = z.infer<
  typeof dynamicEvaluationRunWithEpisodeCountSchema
>;

export const dynamicEvaluationRunEpisodeWithFeedbackSchema = z
  .object({
    episode_id: z.string().uuid(),
    timestamp: z.string().datetime(),
    run_id: z.string().uuid(),
    tags: z.record(z.string(), z.string()),
    task_name: z.string().nullable(),
    // The feedback is given as arrays feedback_metric_names and feedback_values.
    // The arrays are sorted by the metric name.
    feedback_metric_names: z.array(z.string()),
    feedback_values: z.array(z.string()),
  })
  .strict();

export type DynamicEvaluationRunEpisodeWithFeedback = z.infer<
  typeof dynamicEvaluationRunEpisodeWithFeedbackSchema
>;

export const groupedDynamicEvaluationRunEpisodeWithFeedbackSchema = z.object({
  group_key: z.string(),
  ...dynamicEvaluationRunEpisodeWithFeedbackSchema.shape,
});

export type GroupedDynamicEvaluationRunEpisodeWithFeedback = z.infer<
  typeof groupedDynamicEvaluationRunEpisodeWithFeedbackSchema
>;

export const dynamicEvaluationRunStatisticsByMetricNameSchema = z.object({
  metric_name: z.string(),
  count: z.number(),
  avg_metric: z.number(),
  stdev: z.number().nullable(),
  ci_error: z.number().nullable(),
});

export type DynamicEvaluationRunStatisticsByMetricName = z.infer<
  typeof dynamicEvaluationRunStatisticsByMetricNameSchema
>;

export const dynamicEvaluationProjectSchema = z.object({
  name: z.string(),
  count: z.number(),
  last_updated: z.string().datetime(),
});

export type DynamicEvaluationProject = z.infer<
  typeof dynamicEvaluationProjectSchema
>;
