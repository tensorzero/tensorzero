import { clickhouseClient } from "./client.server";
import type { EvaluationRunInfo } from "./evaluations";

export async function getEvalRunIds(
  eval_name: string,
  limit: number = 100,
  offset: number = 0,
): Promise<EvaluationRunInfo[]> {
  const query = `
    SELECT DISTINCT run_tag.value as eval_run_id, run_tag.variant_name as variant_name
    FROM TagInference AS name_tag
    INNER JOIN TagInference AS run_tag
      ON name_tag.inference_id = run_tag.inference_id
    WHERE name_tag.key = 'tensorzero::eval_name'
      AND name_tag.value = {eval_name:String}
      AND run_tag.key = 'tensorzero::eval_run_id'
    ORDER BY eval_run_id, variant_name
    LIMIT {limit:UInt32}
    OFFSET {offset:UInt32}
    `;

  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      eval_name: eval_name,
      limit: limit,
      offset: offset,
    },
  });

  const rows = await result.json<EvaluationRunInfo>();
  return rows;
}

export async function getEvalResults(eval_run_ids: string[]) {
  const query = `
    SELECT * FROM TagInference
    WHERE key = 'tensorzero::eval_run_id'
    AND value IN {eval_run_ids:Array(String)}
  `;
}
