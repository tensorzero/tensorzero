import { z } from "zod";
import { ModelOptionSchema } from "./model_options";
import { zodResolver } from "@hookform/resolvers/zod";

export const SFTFormValuesSchema = z.object({
  function: z.string(),
  metric: z.string(),
  model: ModelOptionSchema,
  variant: z.string(),
  validationSplitPercent: z.number(),
  maxSamples: z.number(),
  threshold: z.number(),
  jobId: z.string(),
});

export type SFTFormValues = z.infer<typeof SFTFormValuesSchema>;
export const SFTFormValuesResolver = zodResolver(SFTFormValuesSchema);
