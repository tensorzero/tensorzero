import fs from "fs";
import path from "path";
import { z } from "zod";
import type { ChatCompletionConfig } from "./config/variant";
import type { FunctionConfig } from "./config/function";
import type { MetricConfig } from "./config/metric";

export async function get_curated_inferences(
  function_name: string,
  function_config: FunctionConfig,
  metric_name: string,
  metric_config: MetricConfig,
  max_samples?: number,
) {
  const fixturesPath = path.join(
    process.cwd(),
    "fixtures",
    "curated_inferences.json",
  );
  const curatedInferences = JSON.parse(fs.readFileSync(fixturesPath, "utf8"));

  return curatedInferences;
}

type InferenceRow = {
  variant_name: string;
  input: string;
  output: string;
  value: number;
  episode_id: string;
};

export type ParsedChatInferenceRow = Omit<InferenceRow, "input" | "output"> & {
  input: Input;
  output: ContentBlockOutput[];
};

export type ParsedJsonInferenceRow = Omit<InferenceRow, "input" | "output"> & {
  input: Input;
  output: JsonInferenceOutput;
};

export type ParsedInferenceRow =
  | ParsedChatInferenceRow
  | ParsedJsonInferenceRow;

export const roleSchema = z.enum(["user", "assistant"]);
export type Role = z.infer<typeof roleSchema>;

export const textInputMessageContentSchema = z.object({
  type: z.literal("text"),
  value: z.any(), // Value type from Rust maps to any in TS
});

export const toolCallSchema = z
  .object({
    name: z.string(),
    arguments: z.string(),
    id: z.string(),
  })
  .strict();
export type ToolCall = z.infer<typeof toolCallSchema>;

export const toolCallInputMessageContentSchema = z
  .object({
    type: z.literal("tool_call"),
    ...toolCallSchema.shape,
  })
  .strict();

export const toolResultSchema = z
  .object({
    name: z.string(),
    result: z.string(),
    id: z.string(),
  })
  .strict();

export const toolResultInputMessageContentSchema = z
  .object({
    type: z.literal("tool_result"),
    ...toolResultSchema.shape,
  })
  .strict();

export const inputMessageContentSchema = z.discriminatedUnion("type", [
  textInputMessageContentSchema,
  toolCallInputMessageContentSchema,
  toolResultInputMessageContentSchema,
]);
export type InputMessageContent = z.infer<typeof inputMessageContentSchema>;

export const inputMessageSchema = z
  .object({
    role: roleSchema,
    content: z.array(inputMessageContentSchema),
  })
  .strict();
export type InputMessage = z.infer<typeof inputMessageSchema>;

export const inputSchema = z
  .object({
    system: z.any().optional(), // Value type from Rust maps to any in TS
    messages: z.array(inputMessageSchema).default([]),
  })
  .strict();
export type Input = z.infer<typeof inputSchema>;

export const jsonInferenceOutputSchema = z.object({
  raw: z.string(),
  parsed: z.any().optional(),
});

export type JsonInferenceOutput = z.infer<typeof jsonInferenceOutputSchema>;

export const toolCallOutputSchema = z
  .object({
    type: z.literal("tool_call"),
    arguments: z.any().optional(), // Value type from Rust maps to any in TS
    id: z.string(),
    name: z.string().optional(),
    raw_arguments: z.string(),
    raw_name: z.string(),
  })
  .strict();

export type ToolCallOutput = z.infer<typeof toolCallOutputSchema>;
export const textSchema = z
  .object({
    type: z.literal("text"),
    text: z.string(),
  })
  .strict();

export type Text = z.infer<typeof textSchema>;

export const contentBlockOutputSchema = z.discriminatedUnion("type", [
  textSchema,
  toolCallOutputSchema,
]);

export type ContentBlockOutput = z.infer<typeof contentBlockOutputSchema>;
