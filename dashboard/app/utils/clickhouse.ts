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

function getInferenceJoinKey(metric_config: MetricConfig): InferenceJoinKey {
  if ("level" in metric_config) {
    switch (metric_config.level) {
      case "inference":
        return InferenceJoinKey.ID;
      case "episode":
        return InferenceJoinKey.EPISODE_ID;
    }
  } else {
    throw new Error(
      "Metric config level is undefined. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new",
    );
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
  function_name: string,
  function_config: FunctionConfig,
  metric_name: string,
  metric_config: MetricConfig,
): Promise<number | null> {
  const inference_table_name = getInferenceTableName(function_config);
  switch (metric_config.type) {
    case "boolean": {
      const inference_join_key = getInferenceJoinKey(metric_config);
      return countMetricData(
        function_name,
        metric_name,
        inference_table_name,
        inference_join_key,
        "boolean",
      );
    }
    case "float": {
      const inference_join_key = getInferenceJoinKey(metric_config);
      return countMetricData(
        function_name,
        metric_name,
        inference_table_name,
        inference_join_key,
        "float",
      );
    }
    case "demonstration":
      return countDemonstrationDataForFunction(
        function_name,
        inference_table_name,
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
      return countMetricData(
        function_name,
        metric_name,
        inference_table_name,
        inference_join_key,
        "boolean",
        { filterGood: true, maximize: metric_config.optimize === "max" },
      );
    case "float":
      return countMetricData(
        function_name,
        metric_name,
        inference_table_name,
        inference_join_key,
        "float",
        {
          filterGood: true,
          maximize: metric_config.optimize === "max",
          threshold,
        },
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

export async function getCuratedInferences(
  function_name: string,
  function_config: FunctionConfig,
  metric_name: string | null,
  metric_config: MetricConfig | null,
  threshold: number,
  max_samples?: number,
) {
  const inference_table_name = getInferenceTableName(function_config);
  if (!metric_config || !metric_name) {
    return queryAllInferencesForFunction(
      function_name,
      inference_table_name,
      max_samples,
    );
  }
  const inference_join_key = getInferenceJoinKey(metric_config);

  switch (metric_config.type) {
    case "boolean":
      return queryCuratedMetricData(
        function_name,
        metric_name,
        inference_table_name,
        inference_join_key,
        "boolean",
        {
          filterGood: true,
          maximize: metric_config.optimize === "max",
          max_samples,
        },
      );
    case "float":
      return queryCuratedMetricData(
        function_name,
        metric_name,
        inference_table_name,
        inference_join_key,
        "float",
        {
          filterGood: true,
          maximize: metric_config.optimize === "max",
          threshold,
          max_samples,
        },
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

async function queryAllInferencesForFunction(
  function_name: string,
  inference_table_name: InferenceTableName,
  max_samples: number | undefined,
): Promise<ParsedInferenceRow[]> {
  const limitClause = max_samples ? `LIMIT ${max_samples}` : "";
  const query = `SELECT * FROM ${inference_table_name} WHERE function_name = {function_name:String} ${limitClause}`;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { function_name },
  });
  const rows = await resultSet.json<InferenceRow>();
  return parseInferenceRows(rows, inference_table_name);
}

// Generic function to handle both boolean and float metric queries
async function queryCuratedMetricData(
  function_name: string,
  metric_name: string,
  inference_table_name: InferenceTableName,
  inference_join_key: InferenceJoinKey,
  metricType: "boolean" | "float",
  options?: {
    filterGood?: boolean;
    maximize?: boolean;
    threshold?: number;
    max_samples?: number;
  },
): Promise<ParsedInferenceRow[]> {
  const {
    filterGood = false,
    maximize = false,
    threshold,
    max_samples,
  } = options || {};
  const limitClause = max_samples ? `LIMIT ${max_samples}` : "";

  let valueCondition = "";
  if (filterGood) {
    if (metricType === "boolean") {
      valueCondition = `AND value = ${maximize ? 1 : 0}`;
    } else if (metricType === "float" && threshold !== undefined) {
      valueCondition = `AND value ${maximize ? ">" : "<"} {threshold:Float}`;
    }
  }

  const feedbackTable =
    metricType === "boolean" ? "BooleanMetricFeedback" : "FloatMetricFeedback";

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
        ${feedbackTable}
      WHERE
        metric_name = {metric_name:String}
        ${valueCondition}
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
      ...(threshold !== undefined ? { threshold } : {}),
    },
  });
  const rows = await resultSet.json<InferenceRow>();
  return parseInferenceRows(rows, inference_table_name);
}

// Generic function to count metric data
async function countMetricData(
  function_name: string,
  metric_name: string,
  inference_table_name: InferenceTableName,
  inference_join_key: InferenceJoinKey,
  metricType: "boolean" | "float",
  options?: {
    filterGood?: boolean;
    maximize?: boolean;
    threshold?: number;
  },
): Promise<number> {
  const { filterGood = false, maximize = false, threshold } = options || {};

  let valueCondition = "";
  if (filterGood) {
    if (metricType === "boolean") {
      valueCondition = `AND value = ${maximize ? 1 : 0}`;
    } else if (metricType === "float" && threshold !== undefined) {
      valueCondition = `AND value ${maximize ? ">" : "<"} {threshold:Float}`;
    }
  }

  const feedbackTable =
    metricType === "boolean" ? "BooleanMetricFeedback" : "FloatMetricFeedback";

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
        ${feedbackTable}
      WHERE
        metric_name = {metric_name:String}
        ${valueCondition}
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
      ...(threshold !== undefined ? { threshold } : {}),
    },
  });
  const rows = await resultSet.json<{ count: number }>();
  return rows[0].count;
}

async function queryDemonstrationDataForFunction(
  function_name: string,
  inference_table_name: InferenceTableName,
  max_samples: number | undefined,
): Promise<ParsedInferenceRow[]> {
  const limitClause = max_samples ? `LIMIT ${max_samples}` : "";

  const query = `
    SELECT
      i.variant_name,
      i.input,
      f.value as output, -- Since this is a demonstration, the value is the desired output for that inference
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
