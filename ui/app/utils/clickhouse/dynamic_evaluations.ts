import { z } from "zod";

export const dynamicEvaluationRunSchema = z
  .object({
    name: z.string().optional(),
    id: z.string().uuid(),
    variant_pins: z.record(z.string(), z.string()),
    tags: z.record(z.string(), z.string()),
    project_name: z.string(),
    timestamp: z.string().datetime(),
  })
  .strict();

export type DynamicEvaluationRun = z.infer<typeof dynamicEvaluationRunSchema>;

export const dynamicEvaluationRunEpisodeWithFeedbackSchema = z
  .object({
    episode_id: z.string().uuid(),
    timestamp: z.string().datetime(),
    run_id: z.string().uuid(),
    tags: z.record(z.string(), z.string()),
    datapoint_name: z.string().nullable(),
    // The feedback is given as arrays feedback_metric_names and feedback_values.
    // The arrays are sorted by the metric name.
    feedback_metric_names: z.array(z.string()),
    feedback_values: z.array(z.string()),
  })
  .strict();

export type DynamicEvaluationRunEpisodeWithFeedback = z.infer<
  typeof dynamicEvaluationRunEpisodeWithFeedbackSchema
>;
