import { z } from "zod";
import type {
  ContentBlockChatOutput,
  Input,
  JsonInferenceOutput,
  JsonValue,
} from "~/types/tensorzero";

// Schema for the create datapoint form data
export const createDatapointFormSchema = z.object({
  action: z.literal("create"),
  dataset_name: z.string().min(1, "Dataset name is required"),
  function_name: z.string().min(1, "Function name is required"),
  function_type: z.enum(["chat", "json"]),
  input: z.string().transform((str) => JSON.parse(str) as Input),
  output: z
    .string()
    .optional()
    .transform((str) =>
      str
        ? (JSON.parse(str) as ContentBlockChatOutput[] | JsonInferenceOutput)
        : undefined,
    ),
  tags: z
    .string()
    .optional()
    .transform((str) =>
      str ? (JSON.parse(str) as Record<string, string>) : undefined,
    ),
  name: z.string().optional(),
  output_schema: z
    .string()
    .optional()
    .transform((str) => (str ? (JSON.parse(str) as JsonValue) : undefined)),
});

export type CreateDatapointFormData = z.infer<typeof createDatapointFormSchema>;

// Type for the data before serialization
export interface CreateDatapointData {
  dataset_name: string;
  function_name: string;
  function_type: "chat" | "json";
  input: Input;
  output?: ContentBlockChatOutput[] | JsonInferenceOutput;
  tags?: Record<string, string>;
  name?: string;
  output_schema?: JsonValue;
}

/**
 * Serializes create datapoint data to FormData for submission
 */
export function serializeCreateDatapointToFormData(
  data: CreateDatapointData,
): FormData {
  const formData = new FormData();
  formData.append("action", "create");
  formData.append("dataset_name", data.dataset_name);
  formData.append("function_name", data.function_name);
  formData.append("function_type", data.function_type);
  formData.append("input", JSON.stringify(data.input));

  if (data.output !== undefined) {
    formData.append("output", JSON.stringify(data.output));
  }

  if (data.tags !== undefined) {
    formData.append("tags", JSON.stringify(data.tags));
  }

  if (data.name !== undefined) {
    formData.append("name", data.name);
  }

  if (data.output_schema !== undefined) {
    formData.append("output_schema", JSON.stringify(data.output_schema));
  }

  return formData;
}

/**
 * Parses and validates FormData into CreateDatapointFormData
 */
export function parseCreateDatapointFormData(
  formData: FormData,
): CreateDatapointFormData {
  const rawData = {
    action: formData.get("action"),
    dataset_name: formData.get("dataset_name"),
    function_name: formData.get("function_name"),
    function_type: formData.get("function_type"),
    input: formData.get("input"),
    output: formData.get("output") || undefined,
    tags: formData.get("tags") || undefined,
    name: formData.get("name") || undefined,
    output_schema: formData.get("output_schema") || undefined,
  };

  return createDatapointFormSchema.parse(rawData);
}
