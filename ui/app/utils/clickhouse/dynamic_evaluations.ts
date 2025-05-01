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

export const dynamicEvaluationRunEpisodeSchema = z.object({
  episode_id: z.string().uuid(),
  timestamp: z.string().datetime(),
  run_id: z.string().uuid(),
  tags: z.record(z.string(), z.string()),
  datapoint_name: z.string().nullable(),
});

export type DynamicEvaluationRunEpisode = z.infer<
  typeof dynamicEvaluationRunEpisodeSchema
>;
