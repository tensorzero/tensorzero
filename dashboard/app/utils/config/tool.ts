import { z } from "zod";

export const ToolChoiceSchema = z.enum(["none", "auto", "required"]).or(
  z.object({
    specific: z.string(),
  }),
);

export type ToolChoice = z.infer<typeof ToolChoiceSchema>;

export const ToolConfigSchema = z.object({
  description: z.string(),
  parameters: z.string(),
  strict: z.boolean().default(false),
});

export type ToolConfig = z.infer<typeof ToolConfigSchema>;
