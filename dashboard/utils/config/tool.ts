import { z } from "zod";
import { jsonSchema7Validator } from "./types";

export const ToolChoice = z.enum(["none", "auto", "required"]).or(
  z.object({
    specific: z.string(),
  })
);

export type ToolChoice = z.infer<typeof ToolChoice>;

export const ToolConfig = z.object({
  description: z.string(),
  parameters: jsonSchema7Validator,
  strict: z.boolean().default(false),
});

export type ToolConfig = z.infer<typeof ToolConfig>;
