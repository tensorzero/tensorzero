import z from "zod";
import { getInferenceTableName } from "./common";
import type { MetricConfig } from "../config/metric";
import type { FunctionConfig } from "../config/function";
import { clickhouseClient } from "./client.server";

export const timeWindowUnitSchema = z.enum([
  "day",
  "week",
  "month",
  "cumulative",
]);
export type TimeWindowUnit = z.infer<typeof timeWindowUnitSchema>;

export async function getVariantPerformances(params: {
  function_name: string;
  function_config: FunctionConfig;
  metric_name: string;
  metric_config: MetricConfig;
  time_window_unit: TimeWindowUnit;
  variant_name?: string;
}) {
  const {
    function_name,
    function_config,
    metric_name,
    metric_config,
    time_window_unit,
    variant_name,
  } = params;
  if (
    metric_config.type === "comment" ||
    metric_config.type === "demonstration"
  ) {
    return undefined;
  }
  const metric_table_name = (() => {
    switch (metric_config.type) {
      case "float":
        return "FloatMetricFeedback";
      case "boolean":
        return "BooleanMetricFeedback";
    }
  })();

  const inference_table_name = (() => {
    switch (function_config.type) {
      case "chat":
        return "ChatInference";
      case "json":
        return "JsonInference";
    }
  })();

  switch (metric_config.level) {
    case "episode":
      return getEpisodePerformances({
        function_name,
        inference_table_name,
        metric_name,
        metric_table_name,
        time_window_unit,
        variant_name,
      });
    case "inference":
      return getInferencePerformances({
        function_name,
        inference_table_name,
        metric_name,
        metric_table_name,
        time_window_unit,
        variant_name,
      });
  }
}

const variantPerformanceRowSchema = z.object({
  period_start: z.string().datetime(),
  variant_name: z.string(),
  count: z.number(),
  avg_metric: z.number(),
  stdev: z.number().nullable(),
  ci_error: z.number().nullable(),
});
export type VariantPerformanceRow = z.infer<typeof variantPerformanceRowSchema>;

async function getEpisodePerformances(params: {
  function_name: string;
  inference_table_name: string;
  metric_name: string;
  metric_table_name: string;
  time_window_unit: TimeWindowUnit;
  variant_name?: string;
}): Promise<VariantPerformanceRow[]> {
  const {
    function_name,
    inference_table_name,
    metric_name,
    metric_table_name,
    time_window_unit,
    variant_name,
  } = params;

  const variantFilter = variant_name
    ? " AND i.variant_name = {variant_name:String}"
    : "";

  // Different query for cumulative stats
  const query =
    time_window_unit === "cumulative"
      ? `
WITH sub AS (
    SELECT
        i.variant_name AS variant_name,
        i.episode_id AS episode_id,
        any(f.value) AS value_per_episode
    FROM ${inference_table_name} i
    JOIN (
        SELECT
            target_id,
            value,
            ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
        FROM ${metric_table_name}
        WHERE metric_name = {metric_name:String}
    ) f ON i.episode_id = f.target_id AND f.rn = 1
    WHERE
        i.function_name = {function_name:String}${variantFilter}
    GROUP BY
        variant_name,
        episode_id
)
SELECT
    '1970-01-01T00:00:00.000Z' AS period_start,
    variant_name,
    toUInt32(count()) AS count,
    avg(value_per_episode) AS avg_metric,
    stddevSamp(value_per_episode) AS stdev,
    1.96 * (stddevSamp(value_per_episode) / sqrt(count())) AS ci_error
FROM sub
GROUP BY
    variant_name
ORDER BY
    variant_name ASC
`
      : `
-- This query calculates the average value of a metric for
-- each (variant_name, period_start), counting each episode exactly once rather than
-- counting multiple inference episodes.

WITH sub AS (
    /* Round 'timestamp' down to the beginning of the period. */
    SELECT
        dateTrunc({time_window_unit:String}, i.timestamp) AS period_start,

        /* We'll group by variant_name as well, so we have a separate row per variant. */
        i.variant_name AS variant_name,

        /* We'll group by each unique episode_id, so that each episode is counted once. */
        i.episode_id AS episode_id,

        /*
           If there might be multiple ${metric_table_name} rows for the same episode,
           we pick just one value (or an aggregate of values) per episode.
           In this example, we use any(f.value), which means "pick any single row's value."
           Alternatively, you could do something like avg(f.value) if you want to combine
           multiple values per episode.
        */
        any(f.value) AS value_per_episode
    FROM ${inference_table_name} i
    JOIN (
        SELECT
            target_id,
            value,
            ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
        FROM ${metric_table_name}
        WHERE metric_name = {metric_name:String}
    ) f ON i.episode_id = f.target_id AND f.rn = 1
    WHERE
        i.function_name = {function_name:String}${variantFilter}
    GROUP BY
        period_start,
        variant_name,
        episode_id
)
SELECT
    /* The 'period_start' column from the subquery, i.e. dateTrunc({time_window_unit:String}, i.timestamp). */
    formatDateTime(period_start, '%Y-%m-%dT%H:%i:%S.000Z') AS period_start,

    /* The variant_name from the subquery. */
    variant_name,

    /*
       The count() here is effectively the number of unique episodes in that
       (period_start, variant_name) group, because the subquery returns only one
       row per episode_id.
    */
    toUInt32(count()) AS count,
    avg(value_per_episode) AS avg_metric,
    stddevSamp(value_per_episode) AS stdev,
    1.96 * (stddevSamp(value_per_episode) / sqrt(count())) AS ci_error

FROM sub
/*
   Now we group in the *outer* query by (period_start, variant_name)
   to combine all episodes that share the same period_start + variant_name.
*/
GROUP BY
    period_start,
    variant_name
ORDER BY
    period_start ASC,
    variant_name ASC
;
  `;

  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      time_window_unit,
      function_name,
      metric_name,
      ...(variant_name ? { variant_name } : {}),
    },
  });
  const rows = await resultSet.json();
  const parsedRows = z.array(variantPerformanceRowSchema).parse(rows);
  return parsedRows;
}

async function getInferencePerformances(params: {
  function_name: string;
  inference_table_name: string;
  metric_name: string;
  metric_table_name: string;
  time_window_unit: TimeWindowUnit;
  variant_name?: string;
}) {
  const {
    function_name,
    inference_table_name,
    metric_name,
    metric_table_name,
    time_window_unit,
    variant_name,
  } = params;

  const variantFilter = variant_name
    ? " AND i.variant_name = {variant_name:String}"
    : "";

  // Different query for cumulative stats
  const query =
    time_window_unit === "cumulative"
      ? `
SELECT
    '1970-01-01T00:00:00.000Z' AS period_start,
    i.variant_name AS variant_name,
    toUInt32(count()) AS count,
    avg(f.value) AS avg_metric,
    stddevSamp(f.value) AS stdev,
    1.96 * (stddevSamp(f.value) / sqrt(count())) AS ci_error
FROM ${inference_table_name} i
JOIN (
    SELECT
        target_id,
        value,
        ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
    FROM ${metric_table_name}
    WHERE metric_name = {metric_name:String}
) f ON i.id = f.target_id AND f.rn = 1
WHERE
    i.function_name = {function_name:String}${variantFilter}
GROUP BY
    variant_name
ORDER BY
    variant_name ASC
`
      : `
SELECT
    formatDateTime(dateTrunc({time_window_unit:String}, i.timestamp), '%Y-%m-%dT%H:%i:%S.000Z') AS period_start,
    i.variant_name AS variant_name,
    toUInt32(count()) AS count,
    avg(f.value) AS avg_metric,
    stddevSamp(f.value) AS stdev,
    1.96 * (stddevSamp(f.value) / sqrt(count())) AS ci_error
FROM ${inference_table_name} i
JOIN (
    SELECT
        target_id,
        value,
        ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
    FROM ${metric_table_name}
    WHERE metric_name = {metric_name:String}
) f ON i.id = f.target_id AND f.rn = 1
WHERE
    i.function_name = {function_name:String}${variantFilter}
GROUP BY
    period_start,
    variant_name
ORDER BY
    period_start ASC,
    variant_name ASC
  `;

  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      function_name,
      metric_name,
      time_window_unit,
      ...(variant_name ? { variant_name } : {}),
    },
  });

  const rows = await resultSet.json();
  const parsedRows = z.array(variantPerformanceRowSchema).parse(rows);
  return parsedRows;
}

const variantCountsSchema = z.object({
  variant_name: z.string(),
  count: z.number(),
  last_used: z.string().datetime(),
});
export type VariantCounts = z.infer<typeof variantCountsSchema>;

export async function getVariantCounts(params: {
  function_name: string;
  function_config: FunctionConfig;
}): Promise<VariantCounts[]> {
  const { function_name, function_config } = params;
  const inference_table_name = getInferenceTableName(function_config);
  const query = `
SELECT
    variant_name,
    toUInt32(count()) AS count,
    formatDateTime(max(timestamp), '%Y-%m-%dT%H:%i:%S.000Z') AS last_used
FROM ${inference_table_name}
WHERE function_name = {function_name:String}
GROUP BY variant_name
ORDER BY count DESC
`;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { function_name },
  });
  const rows = await resultSet.json();
  const parsedRows = z.array(variantCountsSchema).parse(rows);
  return parsedRows;
}

export async function getUsedVariants(
  function_name: string,
): Promise<string[]> {
  const query = `
  SELECT DISTINCT variant_name
  FROM (
    SELECT variant_name
    FROM ChatInference
    WHERE function_name = {function_name:String}
    UNION ALL
    SELECT variant_name
    FROM JsonInference
    WHERE function_name = {function_name:String}
  )
`;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { function_name },
  });
  const rows = await resultSet.json();

  const parsedRows = z
    .array(z.object({ variant_name: z.string() }))
    .parse(rows)
    .map((row) => row.variant_name);
  return parsedRows;
}
