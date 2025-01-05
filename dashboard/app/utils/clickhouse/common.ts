import { createClient } from "@clickhouse/client";
import { z } from "zod";
import type { FunctionConfig } from "../config/function";

export const clickhouseClient = createClient({
  url: process.env.CLICKHOUSE_URL,
});

export async function checkClickhouseConnection(): Promise<boolean> {
  try {
    const result = await clickhouseClient.ping();
    return result.success;
  } catch {
    return false;
  }
}

export const roleSchema = z.enum(["user", "assistant"]);
export type Role = z.infer<typeof roleSchema>;

export const textInputMessageContentSchema = z.object({
  type: z.literal("text"),
  value: z.any(), // Value type from Rust maps to any in TS
});
export type textInputMessageContent = z.infer<
  typeof textInputMessageContentSchema
>;

export const toolCallSchema = z
  .object({
    name: z.string(),
    arguments: z.string(),
    id: z.string(),
  })
  .strict();
export type toolCall = z.infer<typeof toolCallSchema>;

export const toolCallInputMessageContentSchema = z
  .object({
    type: z.literal("tool_call"),
    ...toolCallSchema.shape,
  })
  .strict();
export type toolCallInputMessageContent = z.infer<
  typeof toolCallInputMessageContentSchema
>;

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
export type toolResultInputMessageContent = z.infer<
  typeof toolResultInputMessageContentSchema
>;

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

export const inferenceRowSchema = z
  .object({
    variant_name: z.string(),
    input: z.string(),
    output: z.string(),
    episode_id: z.string(),
  })
  .strict();
export type InferenceRow = z.infer<typeof inferenceRowSchema>;

export const parsedChatInferenceRowSchema = inferenceRowSchema
  .omit({
    input: true,
    output: true,
  })
  .extend({
    input: inputSchema,
    output: z.array(contentBlockOutputSchema),
  })
  .strict();
export type ParsedChatInferenceRow = z.infer<
  typeof parsedChatInferenceRowSchema
>;

export const parsedJsonInferenceRowSchema = inferenceRowSchema
  .omit({
    input: true,
    output: true,
  })
  .extend({
    input: inputSchema,
    output: jsonInferenceOutputSchema,
  })
  .strict();
export type ParsedJsonInferenceRow = z.infer<
  typeof parsedJsonInferenceRowSchema
>;

export type ParsedInferenceRow =
  | ParsedChatInferenceRow
  | ParsedJsonInferenceRow;

export function parseInferenceRows(
  rows: InferenceRow[],
  tableName: string,
): ParsedChatInferenceRow[] | ParsedJsonInferenceRow[] {
  if (tableName === "ChatInference") {
    return rows.map((row) => ({
      ...row,
      input: inputSchema.parse(JSON.parse(row.input)),
      output: z.array(contentBlockOutputSchema).parse(JSON.parse(row.output)),
    })) as ParsedChatInferenceRow[];
  } else {
    return rows.map((row) => ({
      ...row,
      input: inputSchema.parse(JSON.parse(row.input)),
      output: jsonInferenceOutputSchema.parse(JSON.parse(row.output)),
    })) as ParsedJsonInferenceRow[];
  }
}

export const InferenceTableName = {
  CHAT: "ChatInference",
  JSON: "JsonInference",
} as const;
export type InferenceTableName =
  (typeof InferenceTableName)[keyof typeof InferenceTableName];

export const InferenceJoinKey = {
  ID: "id",
  EPISODE_ID: "episode_id",
} as const;
export type InferenceJoinKey =
  (typeof InferenceJoinKey)[keyof typeof InferenceJoinKey];

export function getInferenceTableName(
  function_config: FunctionConfig,
): InferenceTableName {
  switch (function_config.type) {
    case "chat":
      return InferenceTableName.CHAT;
    case "json":
      return InferenceTableName.JSON;
  }
}

export interface TableBounds {
  first_id: string; // UUIDv7 string
  last_id: string; // UUIDv7 string
}
