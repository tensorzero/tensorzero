import { z } from "zod";

export const workflowEvaluationRunSchema = z
  .object({
    name: z.string().nullable(),
    id: z.string().uuid(),
    variant_pins: z.record(z.string(), z.string()),
    tags: z.record(z.string(), z.string()),
    project_name: z.string().nullable(),
    timestamp: z.string().datetime(),
  })
  .strict();

export type WorkflowEvaluationRun = z.infer<typeof workflowEvaluationRunSchema>;

export const workflowEvaluationRunWithEpisodeCountSchema =
  workflowEvaluationRunSchema.extend({
    num_episodes: z.number().default(0),
  });

export type WorkflowEvaluationRunWithEpisodeCount = z.infer<
  typeof workflowEvaluationRunWithEpisodeCountSchema
>;

export const workflowEvaluationRunEpisodeWithFeedbackSchema = z
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

export type WorkflowEvaluationRunEpisodeWithFeedback = z.infer<
  typeof workflowEvaluationRunEpisodeWithFeedbackSchema
>;

export const groupedWorkflowEvaluationRunEpisodeWithFeedbackSchema = z.object({
  group_key: z.string(),
  ...workflowEvaluationRunEpisodeWithFeedbackSchema.shape,
});

export type GroupedWorkflowEvaluationRunEpisodeWithFeedback = z.infer<
  typeof groupedWorkflowEvaluationRunEpisodeWithFeedbackSchema
>;

export const workflowEvaluationRunStatisticsByMetricNameSchema = z.object({
  metric_name: z.string(),
  count: z.number(),
  avg_metric: z.number(),
  stdev: z.number().nullable(),
  ci_lower: z.number().nullable(),
  ci_upper: z.number().nullable(),
});

export type WorkflowEvaluationRunStatisticsByMetricName = z.infer<
  typeof workflowEvaluationRunStatisticsByMetricNameSchema
>;

export const workflowEvaluationProjectSchema = z.object({
  name: z.string(),
  count: z.number(),
  last_updated: z.string().datetime(),
});

export type WorkflowEvaluationProject = z.infer<
  typeof workflowEvaluationProjectSchema
>;
