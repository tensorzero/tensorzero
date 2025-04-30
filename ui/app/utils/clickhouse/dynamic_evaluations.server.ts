import { clickhouseClient } from "./client.server";
import {
  dynamicEvaluationRunSchema,
  type DynamicEvaluationRun,
} from "./dynamic_evaluations.client";

export async function getDynamicEvaluationRuns(
  page_size: number,
  offset: number,
): Promise<DynamicEvaluationRun[]> {
  const query = `
    SELECT
      run_display_name as name,
      uint_to_uuid(run_id_uint) as id,
      variant_pins,
      tags,
      project_name,
    FROM dynamic_evaluations
    ORDER BY run_id_uint DESC
    LIMIT {page_size:UInt64} OFFSET {offset:UInt64}
    `;
  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      page_size: page_size,
      offset: offset,
    },
  });
  const rows = await result.json<DynamicEvaluationRun[]>();
  return rows.map((row) => dynamicEvaluationRunSchema.parse(row));
}
