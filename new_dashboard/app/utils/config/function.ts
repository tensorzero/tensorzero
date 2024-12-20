import { z } from "zod";
import {
  RawVariantConfigSchema,
  VariantConfigSchema,
  type VariantConfig,
} from "./variant";

// Common schema for both Chat and Json variants
const baseConfigSchema = z.object({
  variants: z.record(z.string(), VariantConfigSchema),
  system_schema: z.string().optional(),
  user_schema: z.string().optional(),
  assistant_schema: z.string().optional(),
});

// Schema for FunctionConfigChat
export const FunctionConfigChatSchema = baseConfigSchema.extend({
  type: z.literal("chat"),
  tools: z.array(z.string()).default([]),
  tool_choice: z.enum(["none", "auto", "any"]).default("none"), // Assuming these are the ToolChoice variants
  parallel_tool_calls: z.boolean().default(false),
});

// Schema for FunctionConfigJson
export const FunctionConfigJsonSchema = baseConfigSchema.extend({
  type: z.literal("json"),
  output_schema: z.string().optional(),
});

// Combined FunctionConfig schema
export const FunctionConfigSchema = z.discriminatedUnion("type", [
  FunctionConfigChatSchema,
  FunctionConfigJsonSchema,
]);

export type FunctionConfigChat = z.infer<typeof FunctionConfigChatSchema>;
export type FunctionConfigJson = z.infer<typeof FunctionConfigJsonSchema>;
export type FunctionConfig = z.infer<typeof FunctionConfigSchema>;

// Common schema for both Chat and Json variants
const rawBaseConfigSchema = z.object({
  variants: z.record(z.string(), RawVariantConfigSchema),
  system_schema: z.string().optional(),
  user_schema: z.string().optional(),
  assistant_schema: z.string().optional(),
});

// Schema for FunctionConfigChat
export const RawFunctionConfigChatSchema = rawBaseConfigSchema.extend({
  type: z.literal("chat"),
  tools: z.array(z.string()).default([]),
  tool_choice: z.enum(["none", "auto", "any"]).default("none"), // Assuming these are the ToolChoice variants
  parallel_tool_calls: z.boolean().default(false),
});

// Schema for FunctionConfigJson
export const RawFunctionConfigJsonSchema = rawBaseConfigSchema.extend({
  type: z.literal("json"),
  output_schema: z.string().optional(),
});

// Combined FunctionConfig schema
export const RawFunctionConfigSchema = z
  .discriminatedUnion("type", [
    RawFunctionConfigChatSchema,
    RawFunctionConfigJsonSchema,
  ])
  .transform((raw) => {
    const config = { ...raw };
    return {
      ...config,
      load: async function (config_path: string): Promise<FunctionConfig> {
        const loadedVariants: Record<string, VariantConfig> = {};
        for (const [key, variant] of Object.entries(config.variants)) {
          loadedVariants[key] = await variant.load(config_path);
        }
        return {
          ...config,
          variants: loadedVariants,
        };
      },
    };
  });

export type RawFunctionConfigChat = z.infer<typeof RawFunctionConfigChatSchema>;
export type RawFunctionConfigJson = z.infer<typeof RawFunctionConfigJsonSchema>;
export type RawFunctionConfig = z.infer<typeof RawFunctionConfigSchema>;
