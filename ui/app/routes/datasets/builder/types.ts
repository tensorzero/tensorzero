import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";

export const DatasetBuilderFormValuesSchema = z.object({
  dataset: z.string(),
  type: z.enum(["chat", "json"]),
  function: z.string().optional(),
  variant: z.string().optional(),
  metric: z
    .string()
    .nullable()
    .refine((val) => val === null || val !== "", {
      message: "Please select a metric or 'None'",
    }),
  metric_type: z.enum(["boolean", "float"]).optional(),
  threshold: z.number().optional(),
  join_demonstrations: z.boolean().default(false),
});

export type DatasetBuilderFormValues = z.infer<
  typeof DatasetBuilderFormValuesSchema
>;

export const DatasetBuilderFormValuesResolver = zodResolver(
  DatasetBuilderFormValuesSchema,
);
