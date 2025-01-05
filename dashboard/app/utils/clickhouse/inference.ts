import z from "node_modules/zod/lib";
import type { TableBounds } from "./common";
import { clickhouseClient, getInferenceTableName } from "./common";
import { data } from "react-router";
import type { FunctionConfig } from "../config/function";

export const inferenceByIdRowSchema = z
  .object({
    id: z.string().uuid(),
    function_name: z.string(),
    variant_name: z.string(),
    episode_id: z.string().uuid(),
    timestamp: z.string().datetime(),
  })
  .strict();

export type InferenceByIdRow = z.infer<typeof inferenceByIdRowSchema>;

/**
 * Query a table of at most `page_size` Inferences from ChatInference or JsonInference that are
 * before the given `before` ID or after the given `after` ID. If `episode_id` is provided,
 * we only return rows from that specific episode.
 *
 * - If `before` and `after` are both not provided, returns the most recent `page_size` Inferences.
 * - If `before` and `after` are both provided, throw an error.
 * - If `before` is provided, returns the most recent `page_size` Inferences before the given `before` ID.
 * - If `after` is provided, returns the earliest `page_size` Inferences after the given `after` ID.
 *
 * All returned data should be ordered by `id` in descending order.
 */
export async function queryInferenceTable(params: {
  page_size: number;
  before?: string; // UUIDv7 string
  after?: string; // UUIDv7 string
}): Promise<InferenceByIdRow[]> {
  const { page_size, before, after } = params;

  if (before && after) {
    throw new Error("Cannot specify both 'before' and 'after' parameters");
  }

  let query = "";
  const query_params: Record<string, string | number> = { page_size };

  if (!before && !after) {
    // No before/after => just the most recent page_size items
    query = `
      SELECT
        id,
        function_name,
        variant_name,
        episode_id,
        function_type,
        UUIDv7ToDateTime(id) AS timestamp
      FROM InferenceById
      ORDER BY toUInt128(id) DESC
      LIMIT {page_size:UInt32}
    `;
  } else if (before) {
    // "Most recent" page_size before a certain ID => descending sort is fine
    query = `
      SELECT
        id,
        function_name,
        variant_name,
        episode_id,
        function_type,
        UUIDv7ToDateTime(id) AS timestamp
      FROM InferenceById
      WHERE
        toUInt128(id) < toUInt128(toUUID({before:String}))
      ORDER BY toUInt128(id) DESC
      LIMIT {page_size:UInt32}
    `;
    query_params.before = before;
  } else if (after) {
    // "Earliest" page_size after a certain ID => first sort ascending, then reorder descending
    // Subselect:
    query = `
      SELECT
        id,
        function_name,
        variant_name,
        episode_id,
        function_type,
        timestamp
      FROM
      (
        SELECT
          id,
          function_name,
          variant_name,
          episode_id,
          function_type,
          UUIDv7ToDateTime(id) AS timestamp
        FROM InferenceById
        WHERE
          toUInt128(id) > toUInt128(toUUID({after:String}))
        ORDER BY toUInt128(id) ASC
        LIMIT {page_size:UInt32}
      )
      ORDER BY toUInt128(id) DESC
    `;
    query_params.after = after;
  }

  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
      query_params,
    });
    const rows = await resultSet.json<InferenceByIdRow>();
    return rows;
  } catch (error) {
    console.error(error);
    throw data("Error querying inference table", { status: 500 });
  }
}

export async function queryInferenceTableBounds(): Promise<TableBounds> {
  const query = `
   SELECT
  (SELECT id FROM InferenceById WHERE toUInt128(id) = (SELECT MIN(toUInt128(id)) FROM InferenceById)) AS first_id,
  (SELECT id FROM InferenceById WHERE toUInt128(id) = (SELECT MAX(toUInt128(id)) FROM InferenceById)) AS last_id
FROM InferenceById
LIMIT 1
  `;

  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
    });
    const rows = await resultSet.json<TableBounds>();
    return rows[0];
  } catch (error) {
    console.error(error);
    throw data("Error querying inference table bounds", { status: 500 });
  }
}

export const episodeByIdSchema = z
  .object({
    episode_id: z.string().uuid(),
    count: z.number().min(1),
    start_time: z.string().datetime(),
    end_time: z.string().datetime(),
    last_inference_id: z.string().uuid(),
  })
  .strict();

export type EpisodeByIdRow = z.infer<typeof episodeByIdSchema>;

/// Query a table of at most `page_size` Episodes that are before the given `before` ID or after the given `after` ID.
/// Important: The ordering is on the last inference in the episode, not the first (we want to show the freshest episodes first)
/// So we should paginate on the last_inference_ids not the Episode IDs
/// If `before` and `after` are both not provided, the query will return the most recent `page_size` Inferences.
/// If `before` and `after` are both provided, we will throw an error.
/// If `before` is provided, the query will return the most recent `page_size` Inferences before the given `before` ID.
/// If `after` is provided, the query will return the earliest `page_size` Inferences after the given `after` ID.
/// All returned data should be ordered by `id` in descending order.
export async function queryEpisodeTable(params: {
  page_size: number;
  before?: string; // UUIDv7 string
  after?: string; // UUIDv7 string
}): Promise<EpisodeByIdRow[]> {
  const { page_size, before, after } = params;
  if (before && after) {
    throw new Error("Cannot specify both 'before' and 'after' parameters");
  }
  let query = "";
  const query_params: Record<string, string | number> = {
    page_size,
  };
  if (!before && !after) {
    // No before/after => just the most recent page_size items
    query = `
      SELECT
        episode_id,
        toUInt32(count(*)) as count,
        min(UUIDv7ToDateTime(id)) as start_time,
        max(UUIDv7ToDateTime(id)) as end_time,
        argMax(id, toUInt128(id)) as last_inference_id
      FROM InferenceByEpisodeId
      GROUP BY episode_id
      ORDER BY toUInt128(last_inference_id) DESC
      LIMIT {page_size:UInt32}
    `;
  } else if (before) {
    query = `
      SELECT
        episode_id,
        toUInt32(count(*)) as count,
        min(UUIDv7ToDateTime(id)) as start_time,
        max(UUIDv7ToDateTime(id)) as end_time,
        argMax(id, toUInt128(id)) as last_inference_id
      FROM InferenceByEpisodeId
      GROUP BY episode_id
      HAVING toUInt128(last_inference_id) < toUInt128(toUUID({before:String}))
      ORDER BY toUInt128(last_inference_id) DESC
      LIMIT {page_size:UInt32}
    `;
    query_params.before = before;
  } else if (after) {
    query = `
      SELECT
        episode_id,
        count,
        start_time,
        end_time,
        last_inference_id
      FROM
      (
        SELECT
          episode_id,
          toUInt32(count(*)) as count,
          min(UUIDv7ToDateTime(id)) as start_time,
          max(UUIDv7ToDateTime(id)) as end_time,
          argMax(id, toUInt128(id)) as last_inference_id
        FROM InferenceByEpisodeId
        GROUP BY episode_id
        HAVING toUInt128(last_inference_id) > toUInt128(toUUID({after:String}))
        ORDER BY toUInt128(last_inference_id) ASC
        LIMIT {page_size:UInt32}
      )
      ORDER BY toUInt128(last_inference_id) DESC
    `;
    query_params.after = after;
  }

  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
      query_params,
    });
    const rows = await resultSet.json<EpisodeByIdRow>();
    const episodeIds = rows.map((episode) => episode.episode_id);
    const uniqueIds = new Set(episodeIds);
    if (uniqueIds.size !== rows.length) {
      console.warn(
        `Found duplicate episode IDs: ${rows.length - uniqueIds.size} duplicates detected`,
      );
    }
    return rows;
  } catch (error) {
    console.error(error);
    throw data("Error querying episode table", { status: 500 });
  }
}

/// NOTE: these are the last inference IDs of the episodes that have the earliest and latest last inferences,
/// i.e. the first and last episodes in the dashboard sort order
/// You should still paginate on the Inference IDs not the Episode IDs
export async function queryEpisodeTableBounds(): Promise<TableBounds> {
  const query = `
    SELECT
     (SELECT id FROM InferenceByEpisodeId WHERE toUInt128(id) = (SELECT MIN(toUInt128(id)) FROM InferenceByEpisodeId)) AS first_id,
     (SELECT id FROM InferenceByEpisodeId WHERE toUInt128(id) = (SELECT MAX(toUInt128(id)) FROM InferenceByEpisodeId)) AS last_id
    FROM InferenceByEpisodeId
    LIMIT 1
  `;
  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
    });
    const rows = await resultSet.json<TableBounds>();
    return rows[0];
  } catch (error) {
    console.error(error);
    throw data("Error querying inference table bounds", { status: 500 });
  }
}

export async function queryInferenceTableByEpisodeId(params: {
  episode_id: string;
  page_size: number;
  before?: string; // UUIDv7 string
  after?: string; // UUIDv7 string
}): Promise<InferenceByIdRow[]> {
  const { episode_id, page_size, before, after } = params;

  if (before && after) {
    throw new Error("Cannot specify both 'before' and 'after' parameters");
  }

  let query = "";
  const query_params: Record<string, string | number> = {
    page_size,
    episode_id,
  };

  if (!before && !after) {
    // No before/after => just the most recent page_size items
    query = `
      SELECT
        id,
        function_name,
        variant_name,
        episode_id,
        function_type,
        UUIDv7ToDateTime(id) AS timestamp
      FROM InferenceByEpisodeId
      WHERE episode_id = {episode_id:String}
      ORDER BY toUInt128(id) DESC
      LIMIT {page_size:UInt32}
    `;
  } else if (before) {
    // "Most recent" page_size before a certain ID => descending sort is fine
    query = `
      SELECT
        id,
        function_name,
        variant_name,
        episode_id,
        function_type,
        UUIDv7ToDateTime(id) AS timestamp
      FROM InferenceByEpisodeId
      WHERE episode_id = {episode_id:String}
        AND toUInt128(id) < toUInt128(toUUID({before:String}))
      ORDER BY toUInt128(id) DESC
      LIMIT {page_size:UInt32}
    `;
    query_params.before = before;
  } else if (after) {
    // "Earliest" page_size after a certain ID => first sort ascending, then reorder descending
    // Subselect:
    query = `
      SELECT
        id,
        function_name,
        variant_name,
        episode_id,
        function_type,
        timestamp
      FROM
      (
        SELECT
          id,
          function_name,
          variant_name,
          episode_id,
          function_type,
          UUIDv7ToDateTime(id) AS timestamp
        FROM InferenceById
        WHERE episode_id = {episode_id:String}
          AND toUInt128(id) > toUInt128(toUUID({after:String}))
        ORDER BY toUInt128(id) ASC
        LIMIT {page_size:UInt32}
      )
      ORDER BY toUInt128(id) DESC
    `;
    query_params.after = after;
  }

  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
      query_params,
    });
    const rows = await resultSet.json<InferenceByIdRow>();
    return rows;
  } catch (error) {
    console.error(error);
    throw data("Error querying inference table", { status: 500 });
  }
}
export async function queryInferenceTableBoundsByEpisodeId(params: {
  episode_id: string;
}): Promise<TableBounds> {
  const { episode_id } = params;
  const query = `
   SELECT
  (SELECT id FROM InferenceById WHERE toUInt128(id) = (SELECT MIN(toUInt128(id)) FROM InferenceById WHERE episode_id = {episode_id:String})) AS first_id,
  (SELECT id FROM InferenceById WHERE toUInt128(id) = (SELECT MAX(toUInt128(id)) FROM InferenceById WHERE episode_id = {episode_id:String})) AS last_id
FROM InferenceById
LIMIT 1
  `;

  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
      query_params: { episode_id },
    });
    const rows = await resultSet.json<TableBounds>();
    return rows[0];
  } catch (error) {
    console.error(error);
    throw data("Error querying inference table bounds", { status: 500 });
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
