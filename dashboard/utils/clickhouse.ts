import { createClient } from "@clickhouse/client";
import { z } from "zod";

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

export async function queryGoodBooleanMetricData(
  function_name: string,
  metric_name: string,
  inference_table_name: InferenceTableName,
  inference_join_key: InferenceJoinKey,
  maximize: boolean,
  max_samples: number | undefined,
): Promise<ParsedInferenceRow[]> {
  const comparison_operator = maximize ? "= 1" : "= 0"; // Changed from "IS TRUE"/"IS FALSE"
  const limitClause = max_samples ? `LIMIT ${max_samples}` : "";

  const query = `
    SELECT
      i.variant_name,
      i.input,
      i.output,
      f.value,
      i.episode_id
    FROM
      ${inference_table_name} i
    JOIN
      (SELECT
        target_id,
        value,
        ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
      FROM
        BooleanMetricFeedback
      WHERE
        metric_name = {metric_name:String}
        AND value ${comparison_operator}
      ) f ON i.${inference_join_key} = f.target_id and f.rn = 1
    WHERE
      i.function_name = {function_name:String}
    ${limitClause}
  `;

  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      metric_name,
      function_name,
    },
  });
  const rows = await resultSet.json<InferenceRow>();
  return parseInferenceRows(rows, inference_table_name);
}

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
