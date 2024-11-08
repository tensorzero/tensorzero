import { z } from "zod";
import { JSONSchema7 } from "json-schema";
import Ajv from "ajv";
// import draft7MetaSchema from "ajv/dist/refs/json-schema-draft-07.json";

const ajv = new Ajv({ allErrors: true });
// ajv.addMetaSchema(draft7MetaSchema);

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

export const jsonModeSchema = z.enum(["off", "on", "strict", "implicit_tool"]);
export type JsonMode = z.infer<typeof jsonModeSchema>;

export const retryConfigSchema = z.object({
  num_retries: z.number().int().nonnegative().default(0),
  max_delay_s: z.number().nonnegative().default(10),
});
