import { createClient } from "@clickhouse/client";

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

export async function queryGoodBooleanMetricData(
  function_name: string,
  metric_name: string,
  inference_table_name: string,
  inference_join_key: string,
  maximize: boolean,
  max_samples: number | undefined,
) {
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
  return resultSet.json();
}
