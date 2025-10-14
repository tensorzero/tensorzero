import { z } from "zod";
import {
  contentBlockChatOutputSchema,
  jsonInferenceOutputSchema,
  displayInputSchema,
  JsonValueSchema,
} from "./common";

/**
 * Schema representing a fully-qualified row in the Chat Inference dataset.
 */
export const ChatInferenceDatapointRowSchema = z
  .object({
    dataset_name: z.string(),
    function_name: z.string(),
    name: z.string().optional(),
    id: z.string().uuid(),
    episode_id: z.string().uuid().nullish(),
    input: z.string(),
    output: z.string().nullish(),
    tool_params: z.string(),
    tags: z.record(z.string(), z.string()).nullish(),
    auxiliary: z.string(),
    is_deleted: z.boolean().default(false),
    updated_at: z.string().datetime().default(new Date().toISOString()),
    staled_at: z.string().datetime().nullish(),
    source_inference_id: z.string().uuid().nullish(),
    is_custom: z.boolean(),
  })
  .strict();
export type ChatInferenceDatapointRow = z.infer<
  typeof ChatInferenceDatapointRowSchema
>;

/**
 * Schema representing a fully-qualified row in the JSON Inference dataset.
 */
export const JsonInferenceDatapointRowSchema = z
  .object({
    dataset_name: z.string(),
    function_name: z.string(),
    name: z.string().optional(),
    id: z.string().uuid(),
    episode_id: z.string().uuid().nullish(),
    input: z.string(),
    output: z.string().nullish(),
    output_schema: z.string(),
    tags: z.record(z.string(), z.string()).nullish(),
    auxiliary: z.string(),
    is_deleted: z.boolean().default(false),
    updated_at: z.string().datetime(),
    staled_at: z.string().datetime().nullish(),
    source_inference_id: z.string().uuid().nullish(),
    is_custom: z.boolean(),
  })
  .strict();
export type JsonInferenceDatapointRow = z.infer<
  typeof JsonInferenceDatapointRowSchema
>;

/**
 * Union schema representing a dataset row, which can be either a Chat or JSON inference row.
 */
export const DatapointRowSchema = z.union([
  ChatInferenceDatapointRowSchema,
  JsonInferenceDatapointRowSchema,
]);
export type DatapointRow = z.infer<typeof DatapointRowSchema>;

export const ParsedChatInferenceDatapointRowSchema =
  ChatInferenceDatapointRowSchema.omit({
    input: true,
    output: true,
    tool_params: true,
  }).extend({
    input: displayInputSchema,
    output: z.array(contentBlockChatOutputSchema).optional(),
    tool_params: z.record(z.string(), JsonValueSchema).optional(),
    is_custom: z.boolean(),
  });
export type ParsedChatInferenceDatapointRow = z.infer<
  typeof ParsedChatInferenceDatapointRowSchema
>;

export const ParsedJsonInferenceDatapointRowSchema =
  JsonInferenceDatapointRowSchema.omit({
    input: true,
    output: true,
    output_schema: true,
  }).extend({
    input: displayInputSchema,
    output: jsonInferenceOutputSchema.optional(),
    output_schema: JsonValueSchema,
    is_custom: z.boolean(),
  });
export type ParsedJsonInferenceDatapointRow = z.infer<
  typeof ParsedJsonInferenceDatapointRowSchema
>;

/**
 * Union schema representing a parsed dataset row, which can be either a Chat or JSON inference row.
 */
export const ParsedDatasetRowSchema = z.union([
  ParsedChatInferenceDatapointRowSchema,
  ParsedJsonInferenceDatapointRowSchema,
]);
export type ParsedDatasetRow = z.infer<typeof ParsedDatasetRowSchema>;
