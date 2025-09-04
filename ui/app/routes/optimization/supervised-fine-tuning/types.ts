import { z } from "zod";
import { ModelOptionSchema } from "./model_options";
import { zodResolver } from "@hookform/resolvers/zod";

export const SFTFormValuesSchema = z.object({
  function: z.string().nonempty("Function is required"),
  metric: z
    .string()
    .nullable()
    .refine((val) => val === null || val !== "", {
      message: "Please select a metric or 'None'",
    }),
  model: ModelOptionSchema,
  variant: z.string().nonempty(),
  validationSplitPercent: z
    .number()
    .min(0, "Validation split percent must be greater than 0")
    .max(100, "Validation split percent must be less than 100"),
  maxSamples: z
    .number()
    .min(10, "You need at least 10 curated inferences to fine-tune a model")
    .optional(),
  threshold: z.union([
    z
      .string()
      .refine((val) => val === "" || /^-?(?:\d+(?:\.\d*)?|\.\d+)?$/.test(val), {
        message: "Must be a valid number",
      }),
    z.number(),
  ]),

  jobId: z.string().nonempty("Job ID is required"),
});

export type SFTFormValues = z.infer<typeof SFTFormValuesSchema>;
export const SFTFormValuesResolver = zodResolver(SFTFormValuesSchema);
