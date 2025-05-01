import { clickhouseClient } from "./client.server";
import {
  dynamicEvaluationRunEpisodeSchema,
  dynamicEvaluationRunSchema,
  type DynamicEvaluationRun,
  type DynamicEvaluationRunEpisode,
} from "./dynamic_evaluations";

export async function getDynamicEvaluationRuns(
  page_size: number,
  offset: number,
  run_id?: string,
): Promise<DynamicEvaluationRun[]> {
  const query = `
    SELECT
      run_display_name as name,
      uint_to_uuid(run_id_uint) as id,
      variant_pins,
      tags,
      project_name,
      formatDateTime(UUIDv7ToDateTime(uint_to_uuid(run_id_uint)), '%Y-%m-%dT%H:%i:%SZ') as timestamp
    FROM DynamicEvaluationRun
    ${run_id ? `WHERE toUInt128(toUUID({run_id:String})) = run_id_uint` : ""}
    ORDER BY run_id_uint DESC
    LIMIT {page_size:UInt64} OFFSET {offset:UInt64}
    `;
  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: {
      page_size,
      offset,
      run_id,
    },
  });
  const rows = await result.json<DynamicEvaluationRun[]>();
  return rows.map((row) => dynamicEvaluationRunSchema.parse(row));
}

export async function countDynamicEvaluationRuns(): Promise<number> {
  const query = `
    SELECT toUInt32(count()) as count FROM DynamicEvaluationRun
  `;
  const result = await clickhouseClient.query({ query, format: "JSONEachRow" });
  const rows = await result.json<{ count: number }>();
  return rows[0].count;
}

export async function getDynamicEvaluationRunEpisodesByRunId(
  page_size: number,
  offset: number,
  run_id: string,
): Promise<DynamicEvaluationRunEpisode[]> {
  const query = `
    SELECT
      uint_to_uuid(episode_id_uint) as episode_id,
      formatDateTime(UUIDv7ToDateTime(uint_to_uuid(episode_id_uint)), '%Y-%m-%dT%H:%i:%SZ') as timestamp,
      uint_to_uuid(run_id_uint) as run_id,
      tags,
      datapoint_name,
    FROM DynamicEvaluationRunEpisodeByRunId
    WHERE toUInt128(toUUID({run_id:String})) = run_id_uint
    ORDER BY episode_id_uint DESC
    LIMIT {page_size:UInt64} OFFSET {offset:UInt64}
  `;
  const result = await clickhouseClient.query({
    query,

    format: "JSONEachRow",
    query_params: { page_size, offset, run_id },
  });

  const rows = await result.json<DynamicEvaluationRunEpisode[]>();
  return rows.map((row) => dynamicEvaluationRunEpisodeSchema.parse(row));
}

export async function countDynamicEvaluationRunEpisodesByRunId(
  run_id: string,
): Promise<number> {
  const query = `SELECT toUInt32(count()) as count
                  FROM DynamicEvaluationRunEpisodeByRunId
                  WHERE toUInt128(toUUID({run_id:String})) = run_id_uint`;
  const result = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { run_id },
  });
  const rows = await result.json<{ count: number }>();
  return rows[0].count;
}
