import { z } from "zod";
import { ModelOptionSchema } from "./model_options";

export const SFTFormValuesSchema = z.object({
  function: z.string(),
  metric: z.string(),
  model: ModelOptionSchema,
  variant: z.string(),
  validationSplitPercent: z.number(),
  maxSamples: z.number(),
  threshold: z.number().optional(),
  jobId: z.string(),
});

export type SFTFormValues = z.infer<typeof SFTFormValuesSchema>;
