import { z } from "zod";
import { VariantConfig } from "./variant";

// Common schema for both Chat and Json variants
const baseConfigSchema = z.object({
  variants: z.record(z.string(), VariantConfig),
  system_schema: z.string().optional(),
  user_schema: z.string().optional(),
  assistant_schema: z.string().optional(),
});

// Schema for FunctionConfigChat
export const FunctionConfigChat = baseConfigSchema.extend({
  type: z.literal("chat"),
  tools: z.array(z.string()).default([]),
  tool_choice: z.enum(["none", "auto", "any"]).default("none"), // Assuming these are the ToolChoice variants
  parallel_tool_calls: z.boolean().default(false),
});

// Schema for FunctionConfigJson
export const FunctionConfigJson = baseConfigSchema.extend({
  type: z.literal("json"),
  output_schema: z.string().optional(),
});

// Combined FunctionConfig schema
export const FunctionConfig = z.discriminatedUnion("type", [
  FunctionConfigChat,
  FunctionConfigJson,
]);

export type FunctionConfigChat = z.infer<typeof FunctionConfigChat>;
export type FunctionConfigJson = z.infer<typeof FunctionConfigJson>;
export type FunctionConfig = z.infer<typeof FunctionConfig>;
