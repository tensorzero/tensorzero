import { z } from "zod";
import type {
  FunctionConfig,
  JsonInferenceOutput,
  JsonValue,
} from "tensorzero-node";

/**
 * JSON types.
 */

export const JsonValueSchema: z.ZodType<JsonValue> = z.lazy(() =>
  z.union([
    z.string(),
    z.number(),
    z.boolean(),
    z.null(),
    z.record(JsonValueSchema),
    z.array(JsonValueSchema),
  ]),
);

export const roleSchema = z.enum(["user", "assistant"]);
export type Role = z.infer<typeof roleSchema>;

export const textInputSchema = z.object({
  type: z.literal("text"),
  value: z.any(), // Value type from Rust maps to any in TS
});
export type TextInput = z.infer<typeof textInputSchema>;

export const templateInputSchema = z.object({
  type: z.literal("template"),
  name: z.string(),
  arguments: z.record(JsonValueSchema.optional()),
});
export type TemplateInput = z.infer<typeof templateInputSchema>;

// The three display text types below handle the scenario
// where the function 1) does not use schemas
export const displayUnstructuredTextInputSchema = z.object({
  type: z.literal("text"),
  text: z.string(),
});
export type DisplayUnstructuredTextInput = z.infer<
  typeof displayUnstructuredTextInputSchema
>;

// 2) uses templates
export const displayTemplateSchema = z.object({
  type: z.literal("template"),
  name: z.string(),
  arguments: z.record(JsonValueSchema.optional()),
});
export type DisplayTemplate = z.infer<typeof displayTemplateSchema>;

// 3) is missing from the config so we don't know
export const displayMissingFunctionTextInputSchema = z.object({
  type: z.literal("missing_function_text"),
  value: z.any(),
});
export type DisplayMissingFunctionTextInput = z.infer<
  typeof displayMissingFunctionTextInputSchema
>;

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

export const thoughtContentSchema = z.object({
  type: z.literal("thought"),
  text: z.string().nullable(),
  signature: z.string().optional(),
  _internal_provider_type: z.string().optional(),
});

export const unknownSchema = z.object({
  type: z.literal("unknown"),
  data: JsonValueSchema,
  model_provider_name: z.string().nullable(),
});
export type Unknown = z.infer<typeof unknownSchema>;

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
  url: z.string().url().nullable(),
  mime_type: z.string(),
});
export type Base64File = z.infer<typeof base64FileSchema>;

export const resolvedBase64FileSchema = z.object({
  dataUrl: z
    .string()
    .url()
    .refine((url) => url.startsWith("data:"), {
      message: "Data URL must start with 'data:'",
    }),
  mime_type: z.string(),
});
export type ResolvedBase64File = z.infer<typeof resolvedBase64FileSchema>;

export const storageKindSchema = z.discriminatedUnion("type", [
  z
    .object({
      type: z.literal("s3_compatible"),
      bucket_name: z.string(),
      region: z.string().nullish(),
      endpoint: z.string().nullish(),
      allow_http: z.boolean().nullish(),
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
  file: base64FileSchema,
  storage_path: storagePathSchema,
  error: z.string(),
});
export type ResolvedImageContentError = z.infer<
  typeof resolvedFileContentErrorSchema
>;

// Types for input to TensorZero
export const inputMessageContentSchema = z.discriminatedUnion("type", [
  textInputSchema,
  templateInputSchema,
  toolCallContentSchema,
  toolResultContentSchema,
  imageContentSchema,
  fileContentSchema,
  rawTextInputSchema,
  thoughtContentSchema,
  unknownSchema,
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
    thoughtContentSchema,
    unknownSchema,
  ],
);
export type ModelInferenceInputMessageContent = z.infer<
  typeof modelInferenceInputMessageContentSchema
>;

export const displayInputMessageContentSchema = z.discriminatedUnion("type", [
  displayUnstructuredTextInputSchema,
  displayTemplateSchema,
  displayMissingFunctionTextInputSchema,
  toolCallContentSchema,
  toolResultContentSchema,
  resolvedFileContentSchema,
  resolvedFileContentErrorSchema,
  rawTextInputSchema,
  thoughtContentSchema,
  unknownSchema,
]);

export type DisplayInputMessageContent = z.infer<
  typeof displayInputMessageContentSchema
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

export const displayModelInferenceInputMessageContentSchema =
  z.discriminatedUnion("type", [
    displayUnstructuredTextInputSchema,
    toolCallContentSchema,
    toolResultContentSchema,
    resolvedFileContentSchema,
    resolvedFileContentErrorSchema,
  ]);

export const displayModelInferenceInputMessageSchema = z.object({
  role: roleSchema,
  content: z.array(displayModelInferenceInputMessageContentSchema),
});
export type DisplayModelInferenceInputMessage = z.infer<
  typeof displayModelInferenceInputMessageSchema
>;

export const displayInputMessageSchema = z
  .object({
    role: roleSchema,
    content: z.array(displayInputMessageContentSchema),
  })
  .strict();
export type DisplayInputMessage = z.infer<typeof displayInputMessageSchema>;

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

export const displayInputSchema = z
  .object({
    system: z.any().optional(), // Value type from Rust maps to any in TS
    messages: z.array(displayInputMessageSchema).default([]),
  })
  .strict();
export type DisplayInput = z.infer<typeof displayInputSchema>;

// Types for main intermediate representations (content blocks and request messages)
export const textContentSchema = z.object({
  type: z.literal("text"),
  text: z.string(),
});
export type TextContent = z.infer<typeof textContentSchema>;

export const contentBlockOutputSchema = z.discriminatedUnion("type", [
  textContentSchema,
  toolCallContentSchema,
  thoughtContentSchema,
  unknownSchema,
]);

export const jsonInferenceOutputSchema = z.object({
  // These fields are explicitly nullable, not undefined.
  raw: z.string().nullable(),
  parsed: JsonValueSchema.nullable(),
}) satisfies z.ZodType<JsonInferenceOutput>;

export const toolCallOutputSchema = z
  .object({
    type: z.literal("tool_call"),
    arguments: JsonValueSchema.nullable(),
    id: z.string(),
    name: z.string().nullable(),
    raw_arguments: z.string(),
    raw_name: z.string(),
  })
  .strict();

export type ToolCallOutput = z.infer<typeof toolCallOutputSchema>;

export const contentBlockChatOutputSchema = z.discriminatedUnion("type", [
  textContentSchema,
  toolCallOutputSchema,
  thoughtContentSchema,
  unknownSchema,
]);

export const modelInferenceOutputContentBlockSchema = z.discriminatedUnion(
  "type",
  [
    textContentSchema,
    toolCallContentSchema,
    thoughtContentSchema,
    unknownSchema,
  ],
);

export type ModelInferenceOutputContentBlock = z.infer<
  typeof modelInferenceOutputContentBlockSchema
>;

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

export const TableBoundsWithCountSchema = TableBoundsSchema.extend({
  count: z.number(),
});
export type TableBoundsWithCount = z.infer<typeof TableBoundsWithCountSchema>;

export const FeedbackBoundsSchema = TableBoundsSchema.extend({
  by_type: z.object({
    boolean: TableBoundsSchema,
    float: TableBoundsSchema,
    demonstration: TableBoundsSchema,
    comment: TableBoundsSchema,
  }),
});
export type FeedbackBounds = z.infer<typeof FeedbackBoundsSchema>;

export const CountSchema = z.object({
  count: z.number(),
});
export type Count = z.infer<typeof CountSchema>;

/**
 * Converts the display input message content to the input message content.
 * This is useful for the case where we've edited a datapoint and need to convert
 * the display form back into something we can write to ClickHouse.
 */

function displayInputMessageContentToInputMessageContent(
  content: DisplayInputMessageContent,
): InputMessageContent {
  switch (content.type) {
    case "text":
      return { type: "text", value: content.text };
    case "missing_function_text":
      return { type: "text", value: content.value };
    case "tool_call":
      return content;
    case "tool_result":
      return content;
    case "file":
      return {
        ...content,
        file: {
          url: content.file.dataUrl,
          mime_type: content.file.mime_type,
        },
        type: "file",
      };
    case "file_error":
      return {
        ...content,
        type: "file",
      };
    case "raw_text":
      return content;
    case "thought":
      return content;
    case "unknown":
      return content;
    case "template":
      return content;
  }
}

/**
 * Converts the display input message to the input message.
 * This is useful for the case where we've edited a datapoint and need to convert
 * the display form back into something we can write to ClickHouse.
 */
function displayInputMessageToInputMessage(
  message: DisplayInputMessage,
): InputMessage {
  return {
    role: message.role,
    content: message.content.map(
      displayInputMessageContentToInputMessageContent,
    ),
  };
}

/**
 * Converts the display input to the input.
 * This is useful for the case where we've edited a datapoint and need to convert
 * the display form back into something we can write to ClickHouse.
 */
export function displayInputToInput(displayInput: DisplayInput): Input {
  return {
    system: displayInput.system,
    messages: displayInput.messages.map(displayInputMessageToInputMessage),
  };
}
