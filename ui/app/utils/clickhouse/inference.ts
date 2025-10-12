import { z } from "zod";
import {
  contentBlockChatOutputSchema,
  jsonInferenceOutputSchema,
} from "./common";
import type {
  DisplayInput,
  DisplayInputMessage,
  ModelInferenceOutputContentBlock,
} from "./common";
import type {
  ContentBlockChatOutput,
  JsonInferenceOutput,
  JsonValue,
  ToolCallConfigDatabaseInsert,
} from "tensorzero-node";

export type ProviderInferenceExtraBody = {
  model_provider_name: string;
  pointer: string;
  value: JsonValue;
};

export type VariantInferenceExtraBody = {
  variant_name: string;
  pointer: string;
  value: JsonValue;
};

export type InferenceExtraBody =
  | ProviderInferenceExtraBody
  | VariantInferenceExtraBody;
export type ParsedChatInferenceRow = {
  id: string;
  function_name: string;
  variant_name: string;
  episode_id: string;
  function_type: "chat";
  input: DisplayInput;
  output: ContentBlockChatOutput[];
  inference_params: Record<string, unknown>;
  processing_time_ms: number;
  timestamp: string;
  tool_params: ToolCallConfigDatabaseInsert | null;
  extra_body: InferenceExtraBody[] | null;
  tags: Record<string, string>;
};

export type ParsedJsonInferenceRow = {
  id: string;
  function_name: string;
  variant_name: string;
  episode_id: string;
  function_type: "json";
  input: DisplayInput;
  output: JsonInferenceOutput;
  inference_params: Record<string, unknown>;
  processing_time_ms: number;
  timestamp: string;
  output_schema: JsonValue;
  extra_body: InferenceExtraBody[] | null;
  tags: Record<string, string>;
};

export type ParsedInferenceRow =
  | ParsedChatInferenceRow
  | ParsedJsonInferenceRow;

export function parseInferenceOutput(
  output: string,
): ContentBlockChatOutput[] | JsonInferenceOutput {
  const parsed = JSON.parse(output);
  if (Array.isArray(parsed)) {
    return z.array(contentBlockChatOutputSchema).parse(parsed);
  }
  return jsonInferenceOutputSchema.parse(parsed);
}

export type ParsedModelInferenceRow = {
  id: string;
  inference_id: string;
  raw_request: string;
  raw_response: string;
  model_name: string;
  model_provider_name: string;
  input_tokens: number | null;
  output_tokens: number | null;
  response_time_ms: number;
  ttft_ms: number | null;
  timestamp: string;
  system: string | null;
  input_messages: DisplayInputMessage[];
  output: ModelInferenceOutputContentBlock[];
  cached: boolean;
  extra_body: string | null;
  tags: Record<string, string>;
};

export type AdjacentIds = {
  previous_id: string | null;
  next_id: string | null;
};
