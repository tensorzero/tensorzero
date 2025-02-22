import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import {
  getComparisonOperator,
  MetricConfigSchema,
} from "~/utils/config/metric";
import { DatasetQueryParamsSchema } from "~/utils/clickhouse/datasets";
import type { DatasetQueryParams } from "~/utils/clickhouse/datasets";
import { getInferenceJoinKey } from "~/utils/clickhouse/curation";

export const DatasetBuilderFormValuesSchema = z.object({
  dataset: z.string().refine((val) => val !== "builder", {
    message: "Dataset name cannot be 'builder'",
  }),
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

export const serializedFormDataToDatasetQueryParams = (
  serializedFormData: string,
): DatasetQueryParams => {
  const parsedData = JSON.parse(serializedFormData);
  const formData = DatasetBuilderFormValuesSchema.parse(parsedData);
  // Build and validate DatasetQueryParams from form data
  const queryParamsResult = DatasetQueryParamsSchema.safeParse({
    inferenceType: formData.type,
    function_name: formData.function,
    variant_name: formData.variant,
    dataset_name: formData.dataset,
    output_source: formData.output_source,
    extra_where: [],
    extra_params: {},
    ...(formData.metric_name && formData.threshold
      ? {
          metric_filter: {
            metric: parsedData.metric_name,
            metric_type: parsedData.metric_config?.type,
            operator: getComparisonOperator(parsedData.metric_config?.optimize),
            threshold: parsedData.threshold,
            join_on: getInferenceJoinKey(parsedData.metric_config?.level),
          },
        }
      : {}),
  });
  if (!queryParamsResult.success) {
    throw new Error(queryParamsResult.error.message);
  }
  return queryParamsResult.data;
};
