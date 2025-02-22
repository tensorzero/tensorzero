import { z } from "zod";
import type { MetricConfig, MetricConfigOptimize } from "../config/metric";
import {
  contentBlockOutputSchema,
  getInferenceTableName,
  InferenceJoinKey,
  InferenceTableName,
  inputSchema,
  jsonInferenceOutputSchema,
} from "./common";
import { clickhouseClient } from "./client.server";
import type { FunctionConfig } from "../config/function";
import { getComparisonOperator } from "../config/metric";
import {
  getInferenceJoinKey,
  type InferenceExample,
  type ParsedChatInferenceExample,
  type ParsedInferenceExample,
  type ParsedJsonInferenceExample,
} from "./curation";

export async function countFeedbacksForMetric(
  function_name: string,
  function_config: FunctionConfig,
  metric_name: string,
  metric_config: MetricConfig,
): Promise<number | null> {
  const inference_table_name = getInferenceTableName(function_config);
  switch (metric_config.type) {
    case "boolean": {
      const inference_join_key = getInferenceJoinKey(metric_config.level);
      return countMetricData(
        function_name,
        metric_name,
        inference_table_name,
        inference_join_key,
        "boolean",
      );
    }
    case "float": {
      const inference_join_key = getInferenceJoinKey(metric_config.level);
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
  switch (metric_config.type) {
    case "boolean":
      return countMetricData(
        function_name,
        metric_name,
        inference_table_name,
        getInferenceJoinKey(metric_config.level),
        "boolean",
        { filterGood: true, optimize: metric_config.optimize },
      );
    case "float":
      return countMetricData(
        function_name,
        metric_name,
        inference_table_name,
        getInferenceJoinKey(metric_config.level),
        "float",
        {
          filterGood: true,
          optimize: metric_config.optimize,
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

  switch (metric_config.type) {
    case "boolean":
      return queryCuratedMetricData(
        function_name,
        metric_name,
        inference_table_name,
        getInferenceJoinKey(metric_config.level),
        "boolean",
        {
          filterGood: true,
          optimize: metric_config.optimize,
          max_samples,
        },
      );
    case "float":
      return queryCuratedMetricData(
        function_name,
        metric_name,
        inference_table_name,
        getInferenceJoinKey(metric_config.level),
        "float",
        {
          filterGood: true,
          optimize: metric_config.optimize,
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
): Promise<ParsedInferenceExample[]> {
  const limitClause = max_samples ? `LIMIT ${max_samples}` : "";
  const query = `SELECT * FROM ${inference_table_name} WHERE function_name = {function_name:String} ${limitClause}`;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { function_name },
  });
  const rows = await resultSet.json<InferenceExample>();
  return parseInferenceExamples(rows, inference_table_name);
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
    optimize?: MetricConfigOptimize;
    threshold?: number;
    max_samples?: number;
  },
): Promise<ParsedInferenceExample[]> {
  const {
    filterGood = false,
    optimize = "max",
    threshold,
    max_samples,
  } = options || {};
  const limitClause = max_samples ? `LIMIT ${max_samples}` : "";

  let valueCondition = "";
  if (filterGood) {
    if (metricType === "boolean") {
      valueCondition = `AND value = ${optimize === "max" ? 1 : 0}`;
    } else if (metricType === "float" && threshold !== undefined) {
      const operator = getComparisonOperator(optimize);
      valueCondition = `AND value ${operator} {threshold:Float}`;
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
  const rows = await resultSet.json<InferenceExample>();
  return parseInferenceExamples(rows, inference_table_name);
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
    optimize?: MetricConfigOptimize;
    threshold?: number;
  },
): Promise<number> {
  const { filterGood = false, optimize = "max", threshold } = options || {};

  let valueCondition = "";
  if (filterGood) {
    if (metricType === "boolean") {
      valueCondition = `AND value = ${optimize === "max" ? 1 : 0}`;
    } else if (metricType === "float" && threshold !== undefined) {
      const operator = getComparisonOperator(optimize);
      valueCondition = `AND value ${operator} {threshold:Float}`;
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
): Promise<ParsedInferenceExample[]> {
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
  const rows = await resultSet.json<InferenceExample>();
  return parseInferenceExamples(rows, inference_table_name);
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

function parseInferenceExamples(
  rows: InferenceExample[],
  tableName: string,
): ParsedChatInferenceExample[] | ParsedJsonInferenceExample[] {
  if (tableName === "ChatInference") {
    return rows.map((row) => ({
      ...row,
      input: inputSchema.parse(JSON.parse(row.input)),
      output: z.array(contentBlockOutputSchema).parse(JSON.parse(row.output)),
    })) as ParsedChatInferenceExample[];
  } else {
    return rows.map((row) => ({
      ...row,
      input: inputSchema.parse(JSON.parse(row.input)),
      output: jsonInferenceOutputSchema.parse(JSON.parse(row.output)),
    })) as ParsedJsonInferenceExample[];
  }
}
