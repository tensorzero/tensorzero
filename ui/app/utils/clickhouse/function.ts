import z from "zod";
import { getInferenceTableName } from "./common";
import type {
  FunctionConfig,
  MetricConfig,
  TimeWindow,
} from "~/types/tensorzero";
import { getClickhouseClient } from "./client.server";

function getTimeWindowInMs(timeWindow: TimeWindow): number {
  switch (timeWindow) {
    case "minute":
      return 60 * 1000; // 1 minute in ms
    case "hour":
      return 60 * 60 * 1000; // 1 hour in ms
    case "day":
      return 24 * 60 * 60 * 1000; // 1 day in ms
    case "week":
      return 7 * 24 * 60 * 60 * 1000; // 1 week in ms
    case "month":
      return 30 * 24 * 60 * 60 * 1000; // 1 month (30 days) in ms
    case "cumulative":
      return 365 * 24 * 60 * 60 * 1000; // 1 year in ms (for cumulative)
  }
}

export async function getVariantPerformances(params: {
  function_name: string;
  function_config: FunctionConfig;
  metric_name: string;
  metric_config: MetricConfig;
  time_window_unit: TimeWindow;
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
  time_window_unit: TimeWindow;
  variant_name?: string;
}): Promise<VariantPerformanceRow[] | undefined> {
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

  const resultSet = await getClickhouseClient().query({
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
  return parsedRows.length > 0 ? parsedRows : undefined;
}

async function getInferencePerformances(params: {
  function_name: string;
  inference_table_name: string;
  metric_name: string;
  metric_table_name: string;
  time_window_unit: TimeWindow;
  variant_name?: string;
}): Promise<VariantPerformanceRow[] | undefined> {
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

  const resultSet = await getClickhouseClient().query({
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
  return parsedRows.length > 0 ? parsedRows : undefined;
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
  const resultSet = await getClickhouseClient().query({
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
  const resultSet = await getClickhouseClient().query({
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

const variantThroughputSchema = z.object({
  period_start: z.string().datetime(),
  variant_name: z.string(),
  count: z.number().min(0),
});
export type VariantThroughput = z.infer<typeof variantThroughputSchema>;

export async function getFunctionThroughputByVariant(
  functionName: string,
  timeWindow: TimeWindow,
  maxPeriods: number,
): Promise<VariantThroughput[]> {
  const timeWindowMs = getTimeWindowInMs(timeWindow);
  // Calculate the time delta in milliseconds that we want to look back from the most recent inference
  const timeDeltaMs = (maxPeriods + 1) * timeWindowMs;
  // Convert time delta to UInt128 representation for UUIDv7 arithmetic
  const timeDeltaUInt128 = getTimeDeltaAsUInt128(timeDeltaMs);

  // Different query for cumulative stats
  const query =
    timeWindow === "cumulative"
      ? `
    SELECT
        -- For cumulative, use a fixed period start
        '1970-01-01T00:00:00.000Z' AS period_start,
        i.variant_name AS variant_name,
        -- Count all inferences per variant
        toUInt32(count()) AS count
    FROM InferenceById i
    WHERE i.function_name = {functionName:String}
    GROUP BY variant_name
    -- Order by variant name
    ORDER BY variant_name DESC
`
      : `
    SELECT
        -- Truncate timestamp to period boundaries (day/week/month) and format as ISO string
        formatDateTime(dateTrunc({timeWindow:String}, UUIDv7ToDateTime(tensorzero_uint_to_uuid(i.id_uint))), '%Y-%m-%dT%H:%i:%S.000Z') AS period_start,
        i.variant_name AS variant_name,
        -- Count inferences per (period, variant) combination
        toUInt32(count()) AS count
    FROM InferenceById i
    WHERE i.function_name = {functionName:String}
    -- Filter based on the most recent inference minus the time delta
    AND i.id_uint >= (
        SELECT max(id_uint) - {timeDeltaUInt128:UInt128}
        FROM InferenceById
        WHERE function_name = {functionName:String}
    )
GROUP BY period_start, variant_name
-- Order by most recent periods first, then by variant name
ORDER BY period_start DESC, variant_name DESC
`;

  const resultSet = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: {
      functionName,
      ...(timeWindow !== "cumulative" ? { timeWindow, timeDeltaUInt128 } : {}),
    },
  });
  const rows = await resultSet.json();

  const parsedRows = z.array(variantThroughputSchema).parse(rows);
  return parsedRows;
}

function getTimeDeltaAsUInt128(timeDeltaMs: number): string {
  // In UUIDv7, the timestamp occupies the first 48 bits (6 bytes) of the 128-bit UUID
  // To convert milliseconds to the equivalent UInt128 difference, we need to shift
  // the timestamp by 80 bits (10 bytes) to position it in the most significant bits

  // Convert milliseconds to hex and pad to 12 characters (48 bits)
  const timestampHex = timeDeltaMs.toString(16).padStart(12, "0");

  // Create UInt128 by placing timestamp in most significant 48 bits, rest zeros
  // This results in: tttttttttttt0000000000000000000000 (in hex)
  const uint128Hex = timestampHex + "0000000000000000000000";

  // Convert to decimal string for ClickHouse UInt128 parameter
  return BigInt("0x" + uint128Hex).toString();
}
