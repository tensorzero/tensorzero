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
  validationSplitPercent: castEmptyStringToNull(
    z
      .number({ errorMap: refineNullErrors("Validation split percent") })
      .min(0, "Validation split percent must be greater than 0")
      .max(100, "Validation split percent must be less than 100"),
  ),
  maxSamples: castEmptyStringToNull(
    z
      .number({ errorMap: refineNullErrors("Max samples") })
      .min(10, "You need at least 10 curated inferences to fine-tune a model")
      .optional(),
  ),
  threshold: castEmptyStringToNull(
    z.number({ errorMap: refineNullErrors("Threshold") }),
  ),
  jobId: z.string().nonempty("Job ID is required"),
});

function castEmptyStringToNull(schema: z.ZodType) {
  return z.preprocess((value) => (value === "" ? null : value), schema);
}

function refineNullErrors(fieldName: string): z.ZodErrorMap {
  return (issue, ctx) => {
    if (issue.code === "invalid_type" && issue.received === "null") {
      return { message: `${fieldName} must not be empty` };
    }
    return { message: ctx.defaultError };
  };
}

export type SFTFormValues = z.infer<typeof SFTFormValuesSchema>;
export const SFTFormValuesResolver = zodResolver(SFTFormValuesSchema);
