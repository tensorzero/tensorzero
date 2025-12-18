import type { JsonValue } from "~/types/tensorzero";

/// Validate if a JSON schema itself is valid.
/// TODO: Right now, we only check if it's a valid JSON object. We can probably do better (e.g. `jsonschema` in Rust has meta-validation).
export function validateJsonSchema(
  schema: JsonValue,
): { valid: true } | { valid: false; error: string } {
  // Schema must be valid JSON (which it always is since it's JsonValue)
  // But we still validate it's a proper object for a JSON schema
  if (typeof schema !== "object" || schema === null || Array.isArray(schema)) {
    return {
      valid: false,
      error: "Output schema must be a JSON object.",
    };
  }
  return { valid: true };
}
