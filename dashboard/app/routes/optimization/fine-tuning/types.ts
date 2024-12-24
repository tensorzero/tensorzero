import { z } from "zod";
import { ModelOptionSchema } from "./model_options";
import { zodResolver } from "@hookform/resolvers/zod";

export const SFTFormValuesSchema = z.object({
  function: z.string().nonempty("Function is required"),
  metric: z.string().nonempty("Metric is required"),
  model: ModelOptionSchema,
  variant: z.string().nonempty(),
  validationSplitPercent: z
    .number()
    .min(0, "Validation split percent must be greater than 0")
    .max(100, "Validation split percent must be less than 100"),
  maxSamples: z
    .number()
    .min(10, "Max samples must be greater than 10")
    .optional(),
  threshold: z.number(),
  jobId: z.string().nonempty("Job ID is required"),
});

export type SFTFormValues = z.infer<typeof SFTFormValuesSchema>;
export const SFTFormValuesResolver = zodResolver(SFTFormValuesSchema);
