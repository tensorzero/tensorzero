import { z } from "zod";

export const ToolChoice = z.enum(["none", "auto", "required"]).or(
  z.object({
    specific: z.string(),
  }),
);

export type ToolChoice = z.infer<typeof ToolChoice>;

export const ToolConfig = z.object({
  description: z.string(),
  parameters: z.string(),
  strict: z.boolean().default(false),
});

export type ToolConfig = z.infer<typeof ToolConfig>;
