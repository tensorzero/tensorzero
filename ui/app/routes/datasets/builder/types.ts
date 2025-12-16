import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { InferenceFilterSchema } from "~/components/querybuilder/inference-filter-schema";
import type { InferenceFilter } from "~/types/tensorzero";

export const DatasetBuilderFormValuesSchema = z.object({
  dataset: z.string().refine((val) => val !== "builder", {
    message: "Dataset name cannot be 'builder'",
  }),
  type: z.enum(["chat", "json"]),
  function: z.string(),
  variant_name: z.string().optional(),
  episode_id: z
    .string()
    .uuid("Must be a valid UUID.")
    .optional()
    .or(z.literal("")),
  search_query: z.string().optional(),
  filters: InferenceFilterSchema.optional(),
  output_source: z.enum(["none", "inference", "demonstration"]),
});

export type DatasetBuilderFormValues = z.infer<
  typeof DatasetBuilderFormValuesSchema
> & {
  // Override the filters type to be more specific since InferenceFilterSchema is z.ZodTypeAny
  filters?: InferenceFilter;
};

export const DatasetBuilderFormValuesResolver = zodResolver(
  DatasetBuilderFormValuesSchema,
);
