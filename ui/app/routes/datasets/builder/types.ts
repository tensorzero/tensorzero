import { z, ZodType } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { getComparisonOperator } from "~/utils/config/feedback";

// This MUST be a type-only import; this code lives in the browser, and tensorzero-node should not be browserified.
import type {
  MetricConfig,
  MetricConfigLevel,
  FilterInferencesForDatasetBuilderRequest,
} from "~/types/tensorzero";

const MetricConfigSchema: ZodType<MetricConfig> = z.any();

export const DatasetBuilderFormValuesSchema = z.object({
  dataset: z.string().refine((val) => val !== "builder", {
    message: "Dataset name cannot be 'builder'",
  }),
  type: z.enum(["chat", "json"]),
  function: z.string(),
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

/**
 * Converts serialized DatasetBuilder form data to FilterInferencesForDatasetBuilderRequest for the new backend API
 */
export const formDataToFilterInferencesForDatasetBuilderRequest = (
  serializedFormData: string,
): {
  datasetName: string;
  params: FilterInferencesForDatasetBuilderRequest;
} => {
  const parsedData = JSON.parse(serializedFormData);
  const formData = DatasetBuilderFormValuesSchema.parse(parsedData);

  const params: FilterInferencesForDatasetBuilderRequest = {
    inference_type: formData.type,
    function_name: formData.function,
    variant_name: formData.variant,
    output_source: formData.output_source,
    metric_filter:
      formData.metric_name && formData.threshold !== undefined
        ? {
            metric: parsedData.metric_name,
            metric_type: parsedData.metric_config?.type,
            operator: getComparisonOperator(parsedData.metric_config?.optimize),
            threshold: parsedData.threshold,
            join_on: parsedData.metric_config?.level as MetricConfigLevel,
          }
        : undefined,
  };

  return { datasetName: formData.dataset, params };
};
