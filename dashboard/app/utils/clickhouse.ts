import { createClient } from "@clickhouse/client";
import { z } from "zod";
import type { FunctionConfig } from "./config/function";
import type { MetricConfig } from "./config/metric";

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

function getInferenceTableName(
  function_config: FunctionConfig,
): InferenceTableName {
  switch (function_config.type) {
    case "chat":
      return InferenceTableName.CHAT;
    case "json":
      return InferenceTableName.JSON;
  }
}

function getMetricTableName(metric_config: MetricConfig): string {
  switch (metric_config.type) {
    case "boolean":
      return "BooleanMetricFeedback";
    case "float":
      return "FloatMetricFeedback";
    case "comment":
      return "CommentFeedback";
    case "demonstration":
      return "DemonstrationFeedback";
  }
}

function getInferenceJoinKey(metric_config: MetricConfig): InferenceJoinKey {
  switch (metric_config.level) {
    case "inference":
      return InferenceJoinKey.ID;
    case "episode":
      return InferenceJoinKey.EPISODE_ID;
  }
}

export async function countInferencesForFunction(
  function_name: string,
  function_config: FunctionConfig,
): Promise<number> {
  const inference_table_name = getInferenceTableName(function_config);
  const query = `SELECT COUNT() AS count FROM ${inference_table_name} WHERE function_name = {function_name:String}`;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { function_name },
  });
  const rows = await resultSet.json<{ count: string }>();
  return Number(rows[0].count);
}

export async function countFeedbacksForMetric(
  metric_name: string,
  metric_config: MetricConfig,
): Promise<number> {
  console.log("metric_config", metric_config);
  const metric_table_name = getMetricTableName(metric_config);

  // Special handling for demonstration feedback which doesn't use metric_name
  if (metric_config.type === "demonstration") {
    const query = `SELECT COUNT() AS count FROM ${metric_table_name}`;
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
    });
    const rows = await resultSet.json<{ count: string }>();
    return Number(rows[0].count);
  }

  // Original logic for other metric types
  const query = `SELECT COUNT() AS count FROM ${metric_table_name} WHERE metric_name = {metric_name:String}`;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { metric_name },
  });
  const rows = await resultSet.json<{ count: string }>();
  return Number(rows[0].count);
}

export async function getCuratedInferences(
  function_name: string,
  function_config: FunctionConfig,
  metric_name: string,
  metric_config: MetricConfig,
  threshold: number,
  max_samples?: number,
) {
  const inference_table_name = getInferenceTableName(function_config);
  const inference_join_key = getInferenceJoinKey(metric_config);

  switch (metric_config.type) {
    case "boolean":
      return queryGoodBooleanMetricData(
        function_name,
        metric_name,
        inference_table_name,
        inference_join_key,
        metric_config.optimize === "max",
        max_samples,
      );
    case "float":
      return queryGoodFloatMetricData(
        function_name,
        metric_name,
        inference_table_name,
        inference_join_key,
        metric_config.optimize === "max",
        threshold,
        max_samples,
      );
    case "demonstration":
      return queryDemonstrationDataForFunction(
        function_name,
        inference_table_name,
        max_samples,
      );
    default:
      throw new Error(`Unsupported metric type: ${metric_config.type}`);
  }
}

export async function countCuratedInferences(
  function_name: string,
  function_config: FunctionConfig,
  metric_name: string,
  metric_config: MetricConfig,
  threshold: number,
) {
  const inference_table_name = getInferenceTableName(function_config);
  const inference_join_key = getInferenceJoinKey(metric_config);
  switch (metric_config.type) {
    case "boolean":
      return countGoodBooleanMetricData(
        function_name,
        metric_name,
        inference_table_name,
        inference_join_key,
        metric_config.optimize === "max",
      );
    case "float":
      return countGoodFloatMetricData(
        function_name,
        metric_name,
        inference_table_name,
        inference_join_key,
        metric_config.optimize === "max",
        threshold,
      );
    case "demonstration":
      return countDemonstrationDataForFunction(
        function_name,
        inference_table_name,
      );
    default:
      throw new Error(`Unsupported metric type: ${metric_config.type}`);
  }
}

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

export async function queryGoodFloatMetricData(
  function_name: string,
  metric_name: string,
  inference_table_name: InferenceTableName,
  inference_join_key: InferenceJoinKey,
  maximize: boolean,
  threshold: number,
  max_samples: number | undefined,
): Promise<ParsedInferenceRow[]> {
  const comparison_operator = maximize ? ">" : "<";
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
        FloatMetricFeedback
      WHERE
        metric_name = {metric_name:String}
        AND value ${comparison_operator} {threshold:Float}
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
      threshold,
    },
  });
  const rows = await resultSet.json<InferenceRow>();
  return parseInferenceRows(rows, inference_table_name);
}

export async function queryDemonstrationDataForFunction(
  function_name: string,
  inference_table_name: InferenceTableName,
  max_samples: number | undefined,
): Promise<ParsedInferenceRow[]> {
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
        inference_id,
        value,
        ROW_NUMBER() OVER (PARTITION BY inference_id ORDER BY timestamp DESC) as rn
      FROM
        DemonstrationFeedback
      ) f ON i.id = f.inference_id and f.rn = 1
    WHERE
      i.function_name = {function_name:String}
    ${limitClause}
  `;

  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      function_name,
    },
  });
  const rows = await resultSet.json<InferenceRow>();
  return parseInferenceRows(rows, inference_table_name);
}

export async function countGoodBooleanMetricData(
  function_name: string,
  metric_name: string,
  inference_table_name: InferenceTableName,
  inference_join_key: InferenceJoinKey,
  maximize: boolean,
): Promise<number> {
  const comparison_operator = maximize ? "= 1" : "= 0"; // Changed from "IS TRUE"/"IS FALSE"

  const query = `
    SELECT
      toUInt32(COUNT(*)) as count
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
  `;

  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      metric_name,
      function_name,
    },
  });
  const rows = await resultSet.json<{ count: number }>();
  return rows[0].count;
}

export async function countGoodFloatMetricData(
  function_name: string,
  metric_name: string,
  inference_table_name: InferenceTableName,
  inference_join_key: InferenceJoinKey,
  maximize: boolean,
  threshold: number,
): Promise<number> {
  const comparison_operator = maximize ? ">" : "<";

  const query = `
    SELECT
      toUInt32(COUNT(*)) as count
    FROM
      ${inference_table_name} i
    JOIN
      (SELECT
        target_id,
        value,
        ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
      FROM
        FloatMetricFeedback
      WHERE
        metric_name = {metric_name:String}
        AND value ${comparison_operator} {threshold:Float}
      ) f ON i.${inference_join_key} = f.target_id and f.rn = 1
    WHERE
      i.function_name = {function_name:String}
  `;

  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      metric_name,
      function_name,
      threshold,
    },
  });
  const rows = await resultSet.json<{ count: number }>();
  return rows[0].count;
}

export async function countDemonstrationDataForFunction(
  function_name: string,
  inference_table_name: InferenceTableName,
): Promise<number> {
  const query = `
    SELECT
      toUInt32(COUNT(*)) as count
    FROM
      ${inference_table_name} i
    JOIN
      (SELECT
        inference_id,
        value,
        ROW_NUMBER() OVER (PARTITION BY inference_id ORDER BY timestamp DESC) as rn
      FROM
        DemonstrationFeedback
      ) f ON i.id = f.inference_id and f.rn = 1
    WHERE
      i.function_name = {function_name:String}
  `;

  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      function_name,
    },
  });
  const rows = await resultSet.json<{ count: number }>();
  return rows[0].count;
}
