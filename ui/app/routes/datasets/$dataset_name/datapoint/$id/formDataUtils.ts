import type {
  ContentBlockChatOutput,
  JsonInferenceOutput,
} from "~/types/tensorzero";
import type { StoredInput } from "~/types/tensorzero";

/**
 * Type for a datapoint form submission containing only the editable fields.
 * This matches what's actually submitted in the form.
 */
export type DatapointFormData = {
  dataset_name: string;
  function_name: string;
  id: string;
  episode_id?: string;
  input: StoredInput;
  output?: ContentBlockChatOutput[] | JsonInferenceOutput;
  tags?: Record<string, string>;
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
 * Parses FormData into a DatapointFormData object, handling missing fields gracefully.
 * - Missing fields are treated as undefined where appropriate
 * - JSON fields are parsed from their string representations
 * - No Zod validation is performed; the backend will validate when saving
 */
export function parseDatapointFormData(formData: FormData): DatapointFormData {
  const dataset_name = formData.get("dataset_name") as string;
  const function_name = formData.get("function_name") as string;
  const id = formData.get("id") as string;
  const episode_id = formData.get("episode_id") as string | null;
  const input = JSON.parse(formData.get("input") as string) as StoredInput;
  const outputStr = formData.get("output") as string | null;
  const output = outputStr ? JSON.parse(outputStr) : undefined;
  const tagsStr = formData.get("tags") as string;
  const tags = tagsStr ? JSON.parse(tagsStr) as Record<string, string> : undefined;

  const result: DatapointFormData = {
    dataset_name,
    function_name,
    id,
    input,
  };

  // Add optional fields only if they have values
  if (episode_id) result.episode_id = episode_id;
  if (output !== undefined) result.output = output;
  if (tags) result.tags = tags;

  return result;
}
