import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";

export const DatasetBuilderFormValuesSchema = z.object({
  dataset_name: z.string(),
  type: z.enum(["chat", "json"]),
  function_name: z.string().optional(),
  variant_name: z.string().optional(),
  metric_name: z.string().optional(),
  threshold: z.number().optional(),
  join_demonstrations: z.boolean().default(false),
});

export type DatasetBuilderFormValues = z.infer<
  typeof DatasetBuilderFormValuesSchema
>;

export const DatasetBuilderFormValuesResolver = zodResolver(
  DatasetBuilderFormValuesSchema,
);
