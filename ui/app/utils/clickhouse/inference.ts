import { z } from "zod";
import {
  contentBlockChatOutputSchema,
  inputSchema,
  jsonInferenceOutputSchema,
  displayInputSchema,
  displayModelInferenceInputMessageSchema,
  modelInferenceOutputContentBlockSchema,
  ZodJsonValueSchema,
} from "./common";
import type {
  JsonInferenceOutput,
  ContentBlockChatOutput,
  Tool,
} from "~/types/tensorzero";

// Note: This schema handles backward compatibility with old database records that don't have
// the 'type' field. The transform ensures all parsed tools have type: "function".
// We use 'as z.ZodType<Tool, z.ZodTypeDef, unknown>' instead of 'satisfies' because:
// - Input type: accepts data with optional 'type' field (old format)
// - Output type: guarantees 'type' field is present (new format)
// This is safe because the transform always adds the 'type' field to the output.
export const toolSchema = z
  .object({
    type: z
      .union([z.literal("function"), z.literal("client_side_function")])
      .optional(),
    description: z.string(),
    parameters: ZodJsonValueSchema,
    name: z.string(),
    strict: z.boolean(),
  })
  .transform((data) => ({
    ...data,
    type: "function" as const,
  })) as z.ZodType<Tool, z.ZodTypeDef, unknown>;

export const toolChoiceSchema = z.union([
  z.literal("none"),
  z.literal("auto"),
  z.literal("required"),
  z.object({ specific: z.string() }),
]);

export const toolCallConfigDatabaseInsertSchema = z.object({
  tools_available: z.array(toolSchema),
  tool_choice: toolChoiceSchema,
  parallel_tool_calls: z.boolean().nullable(),
});

export const providerInferenceExtraBodySchema = z
  .object({
    model_provider_name: z.string(),
    pointer: z.string(),
    value: ZodJsonValueSchema,
  })
  .strict();
export type ProviderInferenceExtraBody = z.infer<
  typeof providerInferenceExtraBodySchema
>;

export const variantInferenceExtraBodySchema = z
  .object({
    variant_name: z.string(),
    pointer: z.string(),
    value: ZodJsonValueSchema,
  })
  .strict();
export type VariantInferenceExtraBody = z.infer<
  typeof variantInferenceExtraBodySchema
>;

export const inferenceExtraBodySchema = z.union([
  providerInferenceExtraBodySchema,
  variantInferenceExtraBodySchema,
]);
export type InferenceExtraBody = z.infer<typeof inferenceExtraBodySchema>;

export const chatInferenceRowSchema = z.object({
  id: z.string().uuid(),
  function_name: z.string(),
  variant_name: z.string(),
  episode_id: z.string().uuid(),
  input: z.string(),
  output: z.string(),
  tool_params: z.string(),
  inference_params: z.string(),
  processing_time_ms: z.number(),
  timestamp: z.string().datetime(),
  extra_body: z.string().nullable(),
  tags: z.record(z.string(), z.string()).default({}),
});

export type ChatInferenceRow = z.infer<typeof chatInferenceRowSchema>;

export const jsonInferenceRowSchema = z.object({
  id: z.string().uuid(),
  function_name: z.string(),
  variant_name: z.string(),
  episode_id: z.string().uuid(),
  input: z.string(),
  output: z.string(),
  output_schema: z.string(),
  inference_params: z.string(),
  processing_time_ms: z.number(),
  timestamp: z.string().datetime(),
  extra_body: z.string().nullable(),
  tags: z.record(z.string(), z.string()).default({}),
});

export type JsonInferenceRow = z.infer<typeof jsonInferenceRowSchema>;

export const inferenceRowSchema = z.discriminatedUnion("function_type", [
  chatInferenceRowSchema.extend({
    function_type: z.literal("chat"),
  }),
  jsonInferenceRowSchema.extend({
    function_type: z.literal("json"),
  }),
]);

export type InferenceRow = z.infer<typeof inferenceRowSchema>;

export const parsedChatInferenceRowSchema = chatInferenceRowSchema
  .omit({
    input: true,
    output: true,
    inference_params: true,
    tool_params: true,
    extra_body: true,
  })
  .extend({
    input: inputSchema,
    output: z.array(contentBlockChatOutputSchema),
    inference_params: z.record(z.string(), z.unknown()),
    tool_params: toolCallConfigDatabaseInsertSchema.nullable(),
    extra_body: z.array(inferenceExtraBodySchema).nullable(),
  });

export type ParsedChatInferenceRow = z.infer<
  typeof parsedChatInferenceRowSchema
>;

export const parsedJsonInferenceRowSchema = jsonInferenceRowSchema
  .omit({
    input: true,
    output: true,
    inference_params: true,
    output_schema: true,
    extra_body: true,
  })
  .extend({
    input: inputSchema,
    output: jsonInferenceOutputSchema,
    inference_params: z.record(z.string(), z.unknown()),
    output_schema: ZodJsonValueSchema,
    extra_body: z.array(inferenceExtraBodySchema).nullable(),
  });

export type ParsedJsonInferenceRow = z.infer<
  typeof parsedJsonInferenceRowSchema
>;

export const parsedInferenceRowSchema = z.discriminatedUnion("function_type", [
  parsedChatInferenceRowSchema.extend({
    function_type: z.literal("chat"),
    input: displayInputSchema,
  }),
  parsedJsonInferenceRowSchema.extend({
    function_type: z.literal("json"),
    input: displayInputSchema,
  }),
]);

export type ParsedInferenceRow = z.infer<typeof parsedInferenceRowSchema>;

export function parseInferenceOutput(
  output: string,
): ContentBlockChatOutput[] | JsonInferenceOutput {
  const parsed = JSON.parse(output);
  if (Array.isArray(parsed)) {
    return z.array(contentBlockChatOutputSchema).parse(parsed);
  }
  return jsonInferenceOutputSchema.parse(parsed);
}

export const modelInferenceRowSchema = z.object({
  id: z.string().uuid(),
  inference_id: z.string().uuid(),
  raw_request: z.string(),
  raw_response: z.string(),
  model_name: z.string(),
  model_provider_name: z.string(),
  input_tokens: z.number().nullable(),
  output_tokens: z.number().nullable(),
  response_time_ms: z.number().nullable(),
  ttft_ms: z.number().nullable(),
  timestamp: z.string().datetime(),
  system: z.string().nullable(),
  input_messages: z.string(),
  output: z.string(),
  cached: z.boolean(),
});

export type ModelInferenceRow = z.infer<typeof modelInferenceRowSchema>;

export const parsedModelInferenceRowSchema = modelInferenceRowSchema
  .omit({
    input_messages: true,
    output: true,
  })
  .extend({
    input_messages: z.array(displayModelInferenceInputMessageSchema),
    output: z.array(modelInferenceOutputContentBlockSchema),
  });

export type ParsedModelInferenceRow = z.infer<
  typeof parsedModelInferenceRowSchema
>;

/// Hacky helper to determine if the output is JSON
// We should continue to refactor our types to avoid stuff like this...
export function isJsonOutput(
  output: ReturnType<typeof parseInferenceOutput>,
): output is JsonInferenceOutput {
  return !Array.isArray(output) && "raw" in output;
}
