import { z } from "zod";
import { ModelOptionSchema } from "./model_options";
import { zodResolver } from "@hookform/resolvers/zod";

const metric = z
  .string()
  .nullable()
  .refine((val) => val === null || val !== "", {
    message: "Please select a metric or 'None'",
  });

const threshold = z.union([
  z
    .string()
    .refine((val) => val === "" || /^-?(?:\d+(?:\.\d*)?|\.\d+)?$/.test(val), {
      message: "Must be a valid number",
    }),
  z.number(),
]);

export const SFTFormValuesSchema = z.object({
  function: z.string().nonempty("Function is required"),
  model: ModelOptionSchema,
  variant: z.string().nonempty(),

  // filters/metrics
  filters: z.array(
    z.object({
      metric,
      threshold,
    }),
  ),
  metric,
  threshold,

  // advanced parameters
  validationSplitPercent: z
    .number()
    .min(0, "Validation split percent must be greater than 0")
    .max(100, "Validation split percent must be less than 100"),
  maxSamples: z
    .number()
    .min(10, "You need at least 10 curated inferences to fine-tune a model")
    .optional(),

  jobId: z.string().nonempty("Job ID is required"),
});

export type SFTFormValues = z.infer<typeof SFTFormValuesSchema>;
export const SFTFormValuesResolver = zodResolver(SFTFormValuesSchema);
