import { z } from "zod";
import type { FunctionConfig, JSONSchema } from "./function";
import { RawVariantConfigSchema } from "./variant.server";
import { ChatCompletionConfigSchema, type VariantConfig } from "./variant";
import path from "path";
import fs from "fs";
import { getUsedVariants } from "../clickhouse/function";

export const DEFAULT_FUNCTION_NAME = "tensorzero::default";

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

// Add this helper function at the top level
async function loadSchemaContent(
  schemaPath: string | undefined,
  configPath: string,
): Promise<JSONSchema | undefined> {
  if (!schemaPath) return undefined;
  const fullPath = path.join(path.dirname(configPath), schemaPath);
  try {
    const content = await fs.promises.readFile(fullPath, "utf-8");
    return JSON.parse(content) as JSONSchema; // Parse and return the JSON
  } catch (error) {
    throw new Error(`Invalid JSON schema in ${fullPath}: ${error}`);
  }
}

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

        const baseConfig = {
          ...config,
          variants: loadedVariants,
          system_schema: config.system_schema
            ? {
                path: config.system_schema,
                content: await loadSchemaContent(
                  config.system_schema,
                  config_path,
                ),
              }
            : undefined,
          user_schema: config.user_schema
            ? {
                path: config.user_schema,
                content: await loadSchemaContent(
                  config.user_schema,
                  config_path,
                ),
              }
            : undefined,
          assistant_schema: config.assistant_schema
            ? {
                path: config.assistant_schema,
                content: await loadSchemaContent(
                  config.assistant_schema,
                  config_path,
                ),
              }
            : undefined,
        };

        if (config.type === "json") {
          return {
            ...baseConfig,
            type: "json" as const,
            output_schema: config.output_schema
              ? {
                  path: config.output_schema,
                  content: await loadSchemaContent(
                    config.output_schema,
                    config_path,
                  ),
                }
              : undefined,
          };
        }

        return {
          ...baseConfig,
          type: "chat" as const,
          tools: config.tools,
          tool_choice: config.tool_choice,
          parallel_tool_calls: config.parallel_tool_calls,
        };
      },
    };
  });

export type RawFunctionConfigChat = z.infer<typeof RawFunctionConfigChatSchema>;
export type RawFunctionConfigJson = z.infer<typeof RawFunctionConfigJsonSchema>;
export type RawFunctionConfig = z.infer<typeof RawFunctionConfigSchema>;

export const getDefaultFunctionWithVariants = async () => {
  const variant_names = await getUsedVariants(DEFAULT_FUNCTION_NAME);
  return {
    type: "chat" as const,
    variants: variant_names.reduce(
      (acc, variant_name) => {
        acc[variant_name] = ChatCompletionConfigSchema.parse({
          type: "chat_completion",
          model: variant_name,
        });
        return acc;
      },
      {} as Record<string, VariantConfig>,
    ),
    tools: [],
    tool_choice: "none" as const,
    parallel_tool_calls: false,
  };
};
