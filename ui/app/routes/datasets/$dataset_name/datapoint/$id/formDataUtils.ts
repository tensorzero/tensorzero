import type { ParsedDatasetRow } from "~/utils/clickhouse/datasets";
import { ParsedDatasetRowSchema } from "~/utils/clickhouse/datasets";
import type {
  ContentBlockChatOutput,
  JsonInferenceOutput,
} from "tensorzero-node";

/**
 * Type for a datapoint with editable fields that can be modified in the UI.
 * This allows output to be null (when editing) while preserving all other fields.
 */
export type DatapointFormData = Omit<ParsedDatasetRow, "output"> & {
  output?: ContentBlockChatOutput[] | JsonInferenceOutput | null;
};

/**
 * Serializes a datapoint object to FormData, handling null and undefined values appropriately.
 * Accepts a ParsedDatasetRow or a DatapointFormData with nullable output field.
 * - undefined and null values are skipped
 * - objects are JSON stringified
 * - primitives are converted to strings
 */
export function serializeDatapointToFormData(
  datapoint: DatapointFormData,
): FormData {
  const formData = new FormData();

  Object.entries(datapoint).forEach(([key, value]) => {
    if (value === undefined || value === null) return;
    if (typeof value === "object") {
      formData.append(key, JSON.stringify(value));
    } else {
      formData.append(key, String(value));
    }
  });

  return formData;
}

/**
 * Parses FormData into a ParsedDatasetRow, handling missing fields gracefully.
 * - Missing fields are treated as null where appropriate
 * - JSON fields are parsed from their string representations
 * - The result is validated against the ParsedDatasetRowSchema
 */
export function parseDatapointFormData(formData: FormData): ParsedDatasetRow {
  const rawData = {
    dataset_name: formData.get("dataset_name"),
    function_name: formData.get("function_name"),
    id: formData.get("id"),
    episode_id: formData.get("episode_id"),
    name: formData.get("name") || undefined,
    input: JSON.parse(formData.get("input") as string),
    output: formData.get("output")
      ? JSON.parse(formData.get("output") as string)
      : undefined,
    output_schema: formData.get("output_schema")
      ? JSON.parse(formData.get("output_schema") as string)
      : undefined,
    tool_params: formData.get("tool_params")
      ? JSON.parse(formData.get("tool_params") as string)
      : undefined,
    tags: JSON.parse(formData.get("tags") as string),
    auxiliary: formData.get("auxiliary"),
    is_deleted: formData.get("is_deleted") === "true",
    updated_at: formData.get("updated_at"),
    staled_at: formData.get("staled_at"),
    source_inference_id: formData.get("source_inference_id"),
    is_custom: formData.get("is_custom") === "true",
  };

  // Only filter out undefined values for union-discriminating fields (output, output_schema, tool_params)
  // Other fields like name should remain even if undefined
  const cleanedData = Object.fromEntries(
    Object.entries(rawData).filter(
      ([key, value]) =>
        value !== undefined ||
        !["output", "output_schema", "tool_params"].includes(key),
    ),
  );
  return ParsedDatasetRowSchema.parse(cleanedData);
}
