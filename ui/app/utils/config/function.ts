import { z } from "zod";
import { VariantConfigSchema } from "./variant";

// Add this interface for schema content
export interface SchemaWithContent {
  path: string;
  content?: JSONSchema;
}

export type JSONSchema = {
  [key: string]: unknown;
};

// Common schema for both Chat and Json variants
const baseConfigSchema = z.object({
  variants: z.record(z.string(), VariantConfigSchema),
  system_schema: z.custom<SchemaWithContent>().optional(),
  user_schema: z.custom<SchemaWithContent>().optional(),
  assistant_schema: z.custom<SchemaWithContent>().optional(),
});

// Schema for FunctionConfigChat
export const FunctionConfigChatSchema = baseConfigSchema.extend({
  type: z.literal("chat"),
  tools: z.array(z.string()).default([]),
  tool_choice: z.enum(["none", "auto", "any"]).default("none"),
  parallel_tool_calls: z.boolean().default(false),
});

// Schema for FunctionConfigJson
export const FunctionConfigJsonSchema = baseConfigSchema.extend({
  type: z.literal("json"),
  output_schema: z.custom<SchemaWithContent>().optional(),
});

// Combined FunctionConfig schema
export const FunctionConfigSchema = z.discriminatedUnion("type", [
  FunctionConfigChatSchema,
  FunctionConfigJsonSchema,
]);

export type FunctionConfigChat = z.infer<typeof FunctionConfigChatSchema>;
export type FunctionConfigJson = z.infer<typeof FunctionConfigJsonSchema>;
export type FunctionConfig = z.infer<typeof FunctionConfigSchema>;
export type FunctionType = FunctionConfig["type"];
