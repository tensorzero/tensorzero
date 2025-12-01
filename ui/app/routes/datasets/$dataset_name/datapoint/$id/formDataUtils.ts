import { z } from "zod";
import type {
  ContentBlockChatOutput,
  JsonInferenceOutput,
  Input,
} from "~/types/tensorzero";

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Safely parses a JSON string with proper error handling.
 * Throws a `ZodError` if JSON parsing fails for consistency with Zod validation.
 */
function safeJsonParse(value: string, fieldName: string) {
  try {
    return JSON.parse(value);
  } catch {
    throw new z.ZodError([
      {
        code: "custom",
        message: `Invalid JSON in ${fieldName} field`,
        path: [fieldName],
      },
    ]);
  }
}

/**
 * Schema for `"delete"` form action.
 */
export const DeleteDatapointFormDataSchema = z.object({
  dataset_name: z.string().min(1, "Dataset name is required"),
  id: z.string().uuid("Invalid datapoint ID format"),
  action: z.literal("delete"),
});

export type DeleteDatapointFormData = z.infer<
  typeof DeleteDatapointFormDataSchema
>;

/**
 * Schema for `"update"` form action.
 */
export const UpdateDatapointFormDataSchema = z.object({
  dataset_name: z.string().min(1, "Dataset name is required"),
  function_name: z.string().min(1, "Function name is required"),
  id: z.string().uuid("Invalid datapoint ID format"),
  episode_id: z.string().uuid("Invalid episode ID format").nullish(),
  input: z.custom<Input>((val) => val !== null && typeof val === "object", {
    message: "Input must be a valid object",
  }),
  // TODO: this could be discriminated over `type` to be more type-safe but it's progress...
  output: z
    .custom<
      ContentBlockChatOutput[] | JsonInferenceOutput
    >((val) => val === undefined || (val !== null && typeof val === "object"), { message: "Output must be a valid object or undefined" })
    .optional(),
  tags: z.record(z.string(), z.string()).optional(),
  action: z.literal("update"),
});

export type UpdateDatapointFormData = z.infer<
  typeof UpdateDatapointFormDataSchema
>;

/**
 * Schema for `"rename"` form action.
 */
export const RenameDatapointFormDataSchema = z.object({
  dataset_name: z.string().min(1, "Dataset name is required"),
  id: z.string().uuid("Invalid datapoint ID format"),
  name: z.string(),
  action: z.literal("rename"),
});

export type RenameDatapointFormData = z.infer<
  typeof RenameDatapointFormDataSchema
>;

/**
 * Schema for `"clone"` form action.
 * The datapoint field is a JSON string containing the full datapoint data to clone.
 */
export const CloneDatapointFormDataSchema = z.object({
  target_dataset: z.string().min(1, "Target dataset name is required"),
  datapoint: z.string().min(1, "Datapoint data is required"),
  action: z.literal("clone"),
});

export type CloneDatapointFormData = z.infer<
  typeof CloneDatapointFormDataSchema
>;

/**
 * Discriminated union of all datapoint actions.
 */
export const DatapointActionSchema = z.discriminatedUnion("action", [
  DeleteDatapointFormDataSchema,
  UpdateDatapointFormDataSchema,
  RenameDatapointFormDataSchema,
  CloneDatapointFormDataSchema,
]);

export type DatapointAction = z.infer<typeof DatapointActionSchema>;

/**
 * Parses FormData for `"delete"` action with Zod validation.
 * Throws `ZodError` if validation fails.
 */
export function parseDeleteDatapointFormData(
  formData: FormData,
): DeleteDatapointFormData {
  const rawData = {
    dataset_name: formData.get("dataset_name"),
    id: formData.get("id"),
    action: formData.get("action"),
  };

  return DeleteDatapointFormDataSchema.parse(rawData);
}

/**
 * Parses FormData for `"update"` action with Zod validation.
 * Throws `ZodError` if validation fails.
 */
export function parseUpdateDatapointFormData(
  formData: FormData,
): UpdateDatapointFormData {
  const inputStr = formData.get("input") as string | null;
  const outputStr = formData.get("output") as string | null;
  const tagsStr = formData.get("tags") as string | null;

  const rawData: Record<string, unknown> = {
    dataset_name: formData.get("dataset_name"),
    function_name: formData.get("function_name"),
    id: formData.get("id"),
    action: formData.get("action"),
  };

  // Only add optional fields if they have values
  const episodeId = formData.get("episode_id");
  if (episodeId) {
    rawData.episode_id = episodeId;
  }

  if (inputStr) {
    rawData.input = safeJsonParse(inputStr, "input");
  }

  if (outputStr) {
    rawData.output = safeJsonParse(outputStr, "output");
  }

  if (tagsStr) {
    rawData.tags = safeJsonParse(tagsStr, "tags");
  }

  return UpdateDatapointFormDataSchema.parse(rawData);
}

/**
 * Parses FormData for `"rename"` action with Zod validation.
 * Throws `ZodError` if validation fails.
 */
export function parseRenameDatapointFormData(
  formData: FormData,
): RenameDatapointFormData {
  const rawData = {
    dataset_name: formData.get("dataset_name"),
    id: formData.get("id"),
    name: formData.get("name"),
    action: formData.get("action"),
  };

  return RenameDatapointFormDataSchema.parse(rawData);
}

/**
 * Parses FormData using discriminated union to determine action type.
 * Throws `ZodError` if validation fails.
 */
export function parseDatapointAction(formData: FormData): DatapointAction {
  const action = formData.get("action");

  // Build raw data object based on common and action-specific fields
  const baseData: Record<string, unknown> = {
    dataset_name: formData.get("dataset_name"),
    id: formData.get("id"),
    action,
  };

  let rawData: Record<string, unknown>;
  if (action === "delete") {
    rawData = baseData;
  } else if (action === "update") {
    const inputStr = formData.get("input") as string | null;
    const outputStr = formData.get("output") as string | null;
    const tagsStr = formData.get("tags") as string | null;

    rawData = {
      ...baseData,
      function_name: formData.get("function_name"),
    };

    // Only add optional fields if they have values
    const episodeId = formData.get("episode_id");
    if (episodeId) {
      rawData.episode_id = episodeId;
    }

    if (inputStr) {
      rawData.input = safeJsonParse(inputStr, "input");
    }

    if (outputStr) {
      rawData.output = safeJsonParse(outputStr, "output");
    }

    if (tagsStr) {
      rawData.tags = safeJsonParse(tagsStr, "tags");
    }
  } else if (action === "rename") {
    rawData = {
      ...baseData,
      name: formData.get("name"),
    };
  } else if (action === "clone") {
    rawData = {
      target_dataset: formData.get("target_dataset"),
      datapoint: formData.get("datapoint"),
      action,
    };
  } else {
    rawData = baseData;
  }

  return DatapointActionSchema.parse(rawData);
}

/**
 * Serializes `"delete"` action data to FormData with validation.
 */
export function serializeDeleteDatapointToFormData(
  data: Omit<DeleteDatapointFormData, "action">,
): FormData {
  const validatedData = DeleteDatapointFormDataSchema.parse({
    ...data,
    action: "delete",
  });

  const formData = new FormData();
  formData.append("dataset_name", validatedData.dataset_name);
  formData.append("id", validatedData.id);
  formData.append("action", validatedData.action);

  return formData;
}

/**
 * Serializes `"update"` action data to FormData with validation.
 */
export function serializeUpdateDatapointToFormData(
  data: Omit<UpdateDatapointFormData, "action">,
): FormData {
  const validatedData = UpdateDatapointFormDataSchema.parse({
    ...data,
    action: "update",
  });

  const formData = new FormData();
  formData.append("dataset_name", validatedData.dataset_name);
  formData.append("function_name", validatedData.function_name);
  formData.append("id", validatedData.id);
  formData.append("input", JSON.stringify(validatedData.input));

  if (validatedData.episode_id) {
    formData.append("episode_id", validatedData.episode_id);
  }
  if (validatedData.output !== undefined) {
    formData.append("output", JSON.stringify(validatedData.output));
  }
  if (validatedData.tags) {
    formData.append("tags", JSON.stringify(validatedData.tags));
  }
  formData.append("action", validatedData.action);

  return formData;
}

/**
 * Serializes `"rename"` action data to FormData with validation.
 */
export function serializeRenameDatapointToFormData(
  data: Omit<RenameDatapointFormData, "action">,
): FormData {
  const validatedData = RenameDatapointFormDataSchema.parse({
    ...data,
    action: "rename",
  });

  const formData = new FormData();
  formData.append("dataset_name", validatedData.dataset_name);
  formData.append("id", validatedData.id);
  formData.append("name", validatedData.name);
  formData.append("action", validatedData.action);

  return formData;
}
