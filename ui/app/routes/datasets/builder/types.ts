import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { MetricConfigSchema } from "~/utils/config/metric";

export const DatasetBuilderFormValuesSchema = z.object({
  dataset: z.string(),
  type: z.enum(["chat", "json"]),
  function: z.string().optional(),
  variant: z.string().optional(),
  metric_name: z
    .string()
    .nullable()
    .refine((val) => val === null || val !== "", {
      message: "Please select a metric or 'None'",
    }),
  metric_config: MetricConfigSchema.optional(),
  threshold: z.number().optional(),
  output_source: z.enum(["none", "inference", "demonstration"]),
});

export type DatasetBuilderFormValues = z.infer<
  typeof DatasetBuilderFormValuesSchema
>;

export const DatasetBuilderFormValuesResolver = zodResolver(
  DatasetBuilderFormValuesSchema,
);
