import { z } from "zod";
import type { FunctionConfig } from "../config/function";

import { createClient } from "@clickhouse/client";

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

export const textInputSchema = z.object({
  type: z.literal("text"),
  value: z.any(), // Value type from Rust maps to any in TS
});
export type TextInput = z.infer<typeof textInputSchema>;

export const toolCallSchema = z
  .object({
    name: z.string(),
    arguments: z.string(),
    id: z.string(),
  })
  .strict();
export type ToolCall = z.infer<typeof toolCallSchema>;

export const toolCallContentSchema = z
  .object({
    type: z.literal("tool_call"),
    ...toolCallSchema.shape,
  })
  .strict();
export type ToolCallContent = z.infer<typeof toolCallContentSchema>;

export const toolResultSchema = z
  .object({
    name: z.string(),
    result: z.string(),
    id: z.string(),
  })
  .strict();
export type ToolResult = z.infer<typeof toolResultSchema>;

export const toolResultContentSchema = z
  .object({
    type: z.literal("tool_result"),
    ...toolResultSchema.shape,
  })
  .strict();
export type ToolResultContent = z.infer<typeof toolResultContentSchema>;

// Types for input to TensorZero
export const inputMessageContentSchema = z.discriminatedUnion("type", [
  textInputSchema,
  toolCallContentSchema,
  toolResultContentSchema,
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

// Types for main intermediate representations (content blocks and request messages)
export const textContentSchema = z.object({
  type: z.literal("text"),
  text: z.string(),
});
export type TextContent = z.infer<typeof textContentSchema>;

export const contentBlockSchema = z.discriminatedUnion("type", [
  textContentSchema,
  toolCallContentSchema,
  toolResultContentSchema,
]);
export type ContentBlock = z.infer<typeof contentBlockSchema>;

export const requestMessageSchema = z.object({
  role: roleSchema,
  content: z.array(contentBlockSchema),
});
export type RequestMessage = z.infer<typeof requestMessageSchema>;

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

export const contentBlockOutputSchema = z.discriminatedUnion("type", [
  textContentSchema,
  toolCallOutputSchema,
]);

export type ContentBlockOutput = z.infer<typeof contentBlockOutputSchema>;

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
  first_id?: string; // UUIDv7 string
  last_id?: string; // UUIDv7 string
}
