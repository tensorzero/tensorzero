import z from "zod";
import { clickhouseClient } from "./common";
import type { MetricConfig } from "../config/metric";
import type { FunctionConfig } from "../config/function";

export const timeWindowUnitSchema = z.enum([
  "hour",
  "day",
  "week",
  "month",
  "quarter",
  "year",
]);
export type TimeWindowUnit = z.infer<typeof timeWindowUnitSchema>;

export async function getVariantPerformances(params: {
  function_name: string;
  function_config: FunctionConfig;
  metric_name: string;
  metric_config: MetricConfig;
  time_window_unit: TimeWindowUnit;
}) {
  const {
    function_name,
    function_config,
    metric_name,
    metric_config,
    time_window_unit,
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
      });
    case "inference":
      return getInferencePerformances({
        function_name,
        inference_table_name,
        metric_name,
        metric_table_name,
        time_window_unit,
      });
  }
}

const variantPerformanceRowSchema = z.object({
  period_start: z.string().date(),
  variant_name: z.string(),
  count: z.number(),
  avg_metric: z.number(),
  stdev: z.number(),
  ci_lower_95: z.number(),
  ci_upper_95: z.number(),
});
export type VariantPerformanceRow = z.infer<typeof variantPerformanceRowSchema>;

async function getEpisodePerformances(params: {
  function_name: string;
  inference_table_name: string;
  metric_name: string;
  metric_table_name: string;
  time_window_unit: TimeWindowUnit;
}): Promise<VariantPerformanceRow[]> {
  const {
    function_name,
    inference_table_name,
    metric_name,
    metric_table_name,
    time_window_unit,
  } = params;
  const query = `
-- This query calculates the average value of a metric for
-- each (variant_name, period_start), counting each episode exactly once rather than
-- counting multiple inference episodes. It also computes a 95% Wald
-- confidence interval around the mean.

WITH sub AS (
    SELECT
        /* Round 'timestamp' down to the beginning of the period. */
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
    FROM tensorzero.${inference_table_name} i
    JOIN tensorzero.${metric_table_name} f
        ON i.episode_id = f.target_id
    WHERE
        /* Filter for the metric you're interested in. */
        f.metric_name = {metric_name:String}

        /* Filter to only inferences with the correct function_name. */
        AND i.function_name = {function_name:String}
    GROUP BY
        period_start,
        variant_name,
        episode_id
)
SELECT
    /* The 'period_start' column from the subquery, i.e. dateTrunc({time_window_unit:String}, i.timestamp). */
    formatDateTime(period_start, '%Y-%m-%dT%H:%M:%S.000Z') AS period_start,

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
    /*
       Wald 95% confidence interval around the mean, using the standard formula:
         mean ± 1.96 * (stdev / sqrt(n))
       This is approximate and assumes a normal distribution and a sufficiently large n.
    */
    (
        avg(value_per_episode)
        - 1.96 * (stddevSamp(value_per_episode) / sqrt(count()))
    ) AS ci_lower_95,
    (
        avg(value_per_episode)
        + 1.96 * (stddevSamp(value_per_episode) / sqrt(count()))
    ) AS ci_upper_95

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
}) {
  const {
    function_name,
    inference_table_name,
    metric_name,
    metric_table_name,
    time_window_unit,
  } = params;
  const query = `
SELECT
    dateTrunc({time_window_unit:String}, i.timestamp) AS period_start,
    i.variant_name AS variant_name,
    toUInt32(count()) AS count,
    avg(f.value) AS avg_metric,
    stddevSamp(f.value) AS stdev,
    /*
       Wald 95% confidence interval around the mean, using the standard formula:
         mean ± 1.96 * (stdev / sqrt(n))
       This is approximate and assumes a normal distribution and a sufficiently large n.
    */
    (
        avg(f.value)
        - 1.96 * (stddevSamp(f.value) / sqrt(count()))
    ) AS ci_lower_95,
    (
        avg(f.value)
        + 1.96 * (stddevSamp(f.value) / sqrt(count()))
    ) AS ci_upper_95
FROM ${inference_table_name} i
JOIN ${metric_table_name} f
    ON i.id = f.target_id
WHERE
    f.metric_name = {metric_name:String}
    AND i.function_name = {function_name:String}
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
    },
  });

  const rows = await resultSet.json();
  console.log(rows);
  const parsedRows = z.array(variantPerformanceRowSchema).parse(rows);
  return parsedRows;
}
