import { z } from "zod";

export const dynamicEvaluationRunSchema = z.object({
  name: z.string(),
  id: z.string().uuid(),
  variant_pins: z.record(z.string(), z.string()),
  tags: z.record(z.string(), z.string()),
  project_name: z.string(),
});

export type DynamicEvaluationRun = z.infer<typeof dynamicEvaluationRunSchema>;
