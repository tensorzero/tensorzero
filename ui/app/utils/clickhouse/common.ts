import { z } from "zod";
import type { FunctionConfig } from "../config/function";

export const roleSchema = z.enum(["user", "assistant"]);
export type Role = z.infer<typeof roleSchema>;

export const textInputSchema = z.object({
  type: z.literal("text"),
  value: z.any(), // Value type from Rust maps to any in TS
});
export type TextInput = z.infer<typeof textInputSchema>;

export const modelInferenceTextInputSchema = z.object({
  type: z.literal("text"),
  text: z.string(),
});
export type ModelInferenceTextInput = z.infer<
  typeof modelInferenceTextInputSchema
>;

export const rawTextInputSchema = z.object({
  type: z.literal("raw_text"),
  value: z.string(),
});
export type RawTextInput = z.infer<typeof rawTextInputSchema>;

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

export const base64FileSchema = z.object({
  url: z.string().nullable(),
  mime_type: z.string(),
});
export type Base64File = z.infer<typeof base64FileSchema>;

export const resolvedBase64FileSchema = z.object({
  url: z.string(),
  mime_type: z.string(),
});
export type ResolvedBase64File = z.infer<typeof resolvedBase64FileSchema>;

export const storageKindSchema = z.discriminatedUnion("type", [
  z
    .object({
      type: z.literal("s3_compatible"),
      bucket_name: z.string(),
      region: z.string().nullable(),
      endpoint: z.string().nullable(),
      allow_http: z.boolean().nullable(),
    })
    .strict(),
  z
    .object({
      type: z.literal("filesystem"),
      path: z.string(),
    })
    .strict(),
  z
    .object({
      type: z.literal("disabled"),
    })
    .strict(),
]);
export type StorageKind = z.infer<typeof storageKindSchema>;

export const storagePathSchema = z.object({
  kind: storageKindSchema,
  path: z.string(),
});
export type StoragePath = z.infer<typeof storagePathSchema>;

export const imageContentSchema = z.object({
  type: z.literal("image"),
  image: base64FileSchema,
  storage_path: storagePathSchema,
});
// Legacy 'image' content block stored in the database
// All new images are written out with the 'file' content block type
export type ImageContent = z.infer<typeof imageContentSchema>;

export const fileContentSchema = z.object({
  type: z.literal("file"),
  file: base64FileSchema,
  storage_path: storagePathSchema,
});
export type FileContent = z.infer<typeof fileContentSchema>;

export const resolvedFileContentSchema = z.object({
  type: z.literal("file"),
  file: resolvedBase64FileSchema,
  storage_path: storagePathSchema,
});
export type ResolvedFileContent = z.infer<typeof resolvedFileContentSchema>;

export const resolvedFileContentErrorSchema = z.object({
  type: z.literal("file_error"),
  error: z.string(),
});
export type ResolvedImageContentError = z.infer<
  typeof resolvedFileContentErrorSchema
>;

// Types for input to TensorZero
export const inputMessageContentSchema = z.discriminatedUnion("type", [
  textInputSchema,
  toolCallContentSchema,
  toolResultContentSchema,
  imageContentSchema,
  fileContentSchema,
  rawTextInputSchema,
]);
export type InputMessageContent = z.infer<typeof inputMessageContentSchema>;

export const modelInferenceInputMessageContentSchema = z.discriminatedUnion(
  "type",
  [
    modelInferenceTextInputSchema,
    toolCallContentSchema,
    toolResultContentSchema,
    imageContentSchema,
    fileContentSchema,
    rawTextInputSchema,
  ],
);
export type ModelInferenceInputMessageContent = z.infer<
  typeof modelInferenceInputMessageContentSchema
>;

export const resolvedInputMessageContentSchema = z.discriminatedUnion("type", [
  textInputSchema,
  toolCallContentSchema,
  toolResultContentSchema,
  resolvedFileContentSchema,
  resolvedFileContentErrorSchema,
  rawTextInputSchema,
]);

export type ResolvedInputMessageContent = z.infer<
  typeof resolvedInputMessageContentSchema
>;

export const inputMessageSchema = z
  .object({
    role: roleSchema,
    content: z.array(inputMessageContentSchema),
  })
  .strict();
export type InputMessage = z.infer<typeof inputMessageSchema>;

export const modelInferenceInputMessageSchema = z
  .object({
    role: roleSchema,
    content: z.array(modelInferenceInputMessageContentSchema),
  })
  .strict();
export type ModelInferenceInputMessage = z.infer<
  typeof modelInferenceInputMessageSchema
>;

export const resolvedInputMessageSchema = z
  .object({
    role: roleSchema,
    content: z.array(resolvedInputMessageContentSchema),
  })
  .strict();
export type ResolvedInputMessage = z.infer<typeof resolvedInputMessageSchema>;

export const inputSchema = z
  .object({
    system: z.any().optional(), // Value type from Rust maps to any in TS
    messages: z.array(inputMessageSchema).default([]),
  })
  .strict();
export type Input = z.infer<typeof inputSchema>;

export const modelInferenceInputSchema = z
  .object({
    system: z.any().optional(), // Value type from Rust maps to any in TS
    messages: z.array(modelInferenceInputMessageSchema).default([]),
  })
  .strict();
export type ModelInferenceInput = z.infer<typeof modelInferenceInputSchema>;

export const resolvedInputSchema = z
  .object({
    system: z.any().optional(), // Value type from Rust maps to any in TS
    messages: z.array(resolvedInputMessageSchema).default([]),
  })
  .strict();
export type ResolvedInput = z.infer<typeof resolvedInputSchema>;

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
  imageContentSchema,
  fileContentSchema,
  rawTextInputSchema,
]);
export type ContentBlock = z.infer<typeof contentBlockSchema>;

export const requestMessageSchema = z.object({
  role: roleSchema,
  content: z.array(contentBlockSchema),
});
export type RequestMessage = z.infer<typeof requestMessageSchema>;

export const jsonInferenceOutputSchema = z.object({
  raw: z.string().default(""),
  parsed: z.any().nullable(),
});

export type JsonInferenceOutput = z.infer<typeof jsonInferenceOutputSchema>;

export const toolCallOutputSchema = z
  .object({
    type: z.literal("tool_call"),
    arguments: z.any().nullable().default(null),
    id: z.string(),
    name: z.string().nullable().default(null),
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
export const TableBoundsSchema = z.object({
  first_id: z.string().uuid().nullable(), // UUIDv7 string
  last_id: z.string().uuid().nullable(), // UUIDv7 string
});
export type TableBounds = z.infer<typeof TableBoundsSchema>;

export const CountSchema = z.object({
  count: z.number(),
});
export type Count = z.infer<typeof CountSchema>;
