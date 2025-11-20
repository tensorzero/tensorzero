import { z } from "zod";
import type { JSONSchema7 } from "json-schema";
import Ajv from "ajv";

const ajv = new Ajv({ allErrors: true });

export const jsonSchema7Validator = z.custom<JSONSchema7>((val) => {
  try {
    if (typeof val !== "object" || val === null) {
      return false;
    }
    return ajv.validateSchema(val);
  } catch {
    return false;
  }
}, "Must be a valid JSON Schema Draft 7 schema");

export const jsonModeSchema = z.enum(["off", "on", "strict", "tool"]);
export type JsonMode = z.infer<typeof jsonModeSchema>;

export const RetryConfigSchema = z.object({
  num_retries: z.number().int().nonnegative().default(0),
  max_delay_s: z.number().nonnegative().default(10),
});
export type RetryConfig = z.infer<typeof RetryConfigSchema>;
