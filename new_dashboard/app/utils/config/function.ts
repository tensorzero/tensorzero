import { z } from "zod";
import { RawVariantConfig, VariantConfig } from "./variant";

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

// Common schema for both Chat and Json variants
const rawBaseConfigSchema = z.object({
  variants: z.record(z.string(), RawVariantConfig),
  system_schema: z.string().optional(),
  user_schema: z.string().optional(),
  assistant_schema: z.string().optional(),
});

// Schema for FunctionConfigChat
export const RawFunctionConfigChat = rawBaseConfigSchema.extend({
  type: z.literal("chat"),
  tools: z.array(z.string()).default([]),
  tool_choice: z.enum(["none", "auto", "any"]).default("none"), // Assuming these are the ToolChoice variants
  parallel_tool_calls: z.boolean().default(false),
});

// Schema for FunctionConfigJson
export const RawFunctionConfigJson = rawBaseConfigSchema.extend({
  type: z.literal("json"),
  output_schema: z.string().optional(),
});

// Combined FunctionConfig schema
export const RawFunctionConfig = z
  .discriminatedUnion("type", [RawFunctionConfigChat, RawFunctionConfigJson])
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

export type RawFunctionConfigChat = z.infer<typeof RawFunctionConfigChat>;
export type RawFunctionConfigJson = z.infer<typeof RawFunctionConfigJson>;
export type RawFunctionConfig = z.infer<typeof RawFunctionConfig>;
