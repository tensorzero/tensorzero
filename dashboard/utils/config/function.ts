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
  tools: z.array(z.string()).default([]),
  tool_choice: z.enum(["none", "auto", "any"]).default("none"), // Assuming these are the ToolChoice variants
  parallel_tool_calls: z.boolean().default(false),
});

// Schema for FunctionConfigJson
export const FunctionConfigJson = baseConfigSchema.extend({
  output_schema: z.string(),
  implicit_tool_call_config: z.object({
    tools_available: z.array(z.any()), // ToolConfig schema would need to be defined separately
    tool_choice: z.object({
      type: z.literal("specific"),
      value: z.string(),
    }),
    parallel_tool_calls: z.boolean(),
  }),
});

// Combined FunctionConfig schema
export const FunctionConfig = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("chat"),
    config: FunctionConfigChat,
  }),
  z.object({
    type: z.literal("json"),
    config: FunctionConfigJson,
  }),
]);

export type FunctionConfigChat = z.infer<typeof FunctionConfigChat>;
export type FunctionConfigJson = z.infer<typeof FunctionConfigJson>;
export type FunctionConfig = z.infer<typeof FunctionConfig>;
