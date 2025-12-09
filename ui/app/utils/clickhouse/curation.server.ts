import type {
  MetricConfigLevel,
  MetricConfigOptimize,
} from "~/types/tensorzero";
import {
  CountSchema,
  getInferenceTableName,
  InferenceJoinKey,
  InferenceTableName,
} from "./common";
import { getClickhouseClient } from "./client.server";
import type { FunctionConfig, JsonInferenceOutput } from "~/types/tensorzero";
import { getComparisonOperator, type FeedbackConfig } from "../config/feedback";
import { logger } from "~/utils/logger";

function getInferenceJoinKey(level: MetricConfigLevel): InferenceJoinKey {
  switch (level) {
    case "inference":
      return InferenceJoinKey.ID;
    case "episode":
      return InferenceJoinKey.EPISODE_ID;
  }
}

export async function countFeedbacksForMetric(
  function_name: string,
  function_config: FunctionConfig,
  metric_name: string,
  metric_config: FeedbackConfig,
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
    case "comment":
    default:
      throw new Error(`Unsupported metric type: ${metric_config.type}`);
  }
}

export async function countCuratedInferences(
  function_name: string,
  function_config: FunctionConfig,
  metric_name: string,
  metric_config: FeedbackConfig,
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
    case "comment":
    default:
      throw new Error(`Unsupported metric type: ${metric_config.type}`);
  }
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

  const resultSet = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: {
      metric_name,
      function_name,
      ...(threshold !== undefined ? { threshold } : {}),
    },
  });
  const rows = await resultSet.json<{ count: number }>();
  const parsedRows = rows.map((row) => CountSchema.parse(row));
  return parsedRows[0].count;
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

  const resultSet = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: {
      function_name,
    },
  });
  const rows = await resultSet.json<{ count: number }>();
  const parsedRows = rows.map((row) => CountSchema.parse(row));
  return parsedRows[0].count;
}

/**
 * When we first introduced LLM Judges, we included the thinking section in the output.
 * We have since removed it, but we need to handle the old data.
 * So, we transform any old LLM Judge outputs to the new format by removing the thinking section from the
 * parsed and raw outputs.
 */
export function handle_llm_judge_output(output: string) {
  let parsed: JsonInferenceOutput;
  try {
    parsed = JSON.parse(output);
  } catch (e) {
    logger.warn("Error parsing LLM Judge output", e);
    // Don't do anything if the output failed to parse
    return output;
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if (!(parsed as any).parsed) {
    // if the output failed to parse don't do anything
    return output;
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if ((parsed as any).parsed.thinking) {
    // there is a thinking section that needs to be removed in the parsed and raw outputs
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delete (parsed as any).parsed.thinking;
    const output = {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      parsed: (parsed as any).parsed,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      raw: JSON.stringify((parsed as any).parsed),
    };
    return JSON.stringify(output);
  }
  return JSON.stringify(parsed);
}
