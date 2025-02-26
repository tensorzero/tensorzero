import z from "zod";
import {
  type ContentBlockOutput,
  type JsonInferenceOutput,
  type TableBounds,
  TableBoundsSchema,
} from "./common";
import {
  contentBlockOutputSchema,
  contentBlockSchema,
  getInferenceTableName,
  inputSchema,
  jsonInferenceOutputSchema,
  requestMessageSchema,
} from "./common";
import { data } from "react-router";
import type { FunctionConfig } from "../config/function";
import { clickhouseClient } from "./client.server";

export const inferenceByIdRowSchema = z
  .object({
    id: z.string().uuid(),
    function_name: z.string(),
    variant_name: z.string(),
    episode_id: z.string().uuid(),
    function_type: z.enum(["chat", "json"]),
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
  /**
   * Extra WHERE clauses, e.g. ["episode_id = {episode_id:UUID}", "variant_name = {variant:String}"]
   * Use param placeholders if you want to avoid manual string interpolation.
   */
  extraWhere?: string[];
  /**
   * Extra query parameters, mapping placeholders (like "episode_id") => actual values
   */
  extraParams?: Record<string, string | number>;
}): Promise<InferenceByIdRow[]> {
  const { page_size, before, after, extraWhere, extraParams } = params;

  if (before && after) {
    throw new Error("Cannot specify both 'before' and 'after' parameters");
  }

  // We'll build up WHERE clauses incrementally
  const whereClauses: string[] = [];

  // Base query params
  const query_params: Record<string, string | number> = {
    page_size,
  };

  // Add the built-in before/after logic
  if (before) {
    whereClauses.push("id_uint < toUInt128(toUUID({before:String}))");
    query_params.before = before;
  }
  if (after) {
    whereClauses.push("id_uint > toUInt128(toUUID({after:String}))");
    query_params.after = after;
  }

  // Merge in caller-supplied where clauses
  if (extraWhere && extraWhere.length) {
    whereClauses.push(...extraWhere);
  }

  // Merge in caller-supplied params
  if (extraParams) {
    Object.entries(extraParams).forEach(([key, value]) => {
      query_params[key] = value;
    });
  }

  // We'll build the actual WHERE portion here (if any).
  const combinedWhere = whereClauses.length
    ? `WHERE ${whereClauses.join(" AND ")}`
    : "";

  let query: string;
  if (!before && !after) {
    // No "before"/"after" => get the most recent page_size items
    query = `
      SELECT
        uint_to_uuid(id_uint) as id,
        function_name,
        variant_name,
        episode_id,
        function_type,
        formatDateTime(UUIDv7ToDateTime(uint_to_uuid(id_uint)), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
      FROM InferenceById
      ${combinedWhere}
      ORDER BY id_uint DESC
      LIMIT {page_size:UInt32}
    `;
  } else if (before) {
    // "Most recent" page_size before given ID
    query = `
      SELECT
        uint_to_uuid(id_uint) as id,
        function_name,
        variant_name,
        episode_id,
        function_type,
        formatDateTime(UUIDv7ToDateTime(uint_to_uuid(id_uint)), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
      FROM InferenceById
      ${combinedWhere}
      ORDER BY id_uint DESC
      LIMIT {page_size:UInt32}
    `;
  } else {
    // "Earliest" page_size after given ID => subselect ascending, then reorder descending
    query = `
      SELECT
        id,
        function_name,
        variant_name,
        episode_id,
        function_type,
        formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
      FROM
      (
        SELECT
          uint_to_uuid(id_uint) as id,
          id_uint,
          function_name,
          variant_name,
          episode_id,
          function_type,
          formatDateTime(UUIDv7ToDateTime(uint_to_uuid(id_uint)), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
        FROM InferenceById
        ${combinedWhere}
        ORDER BY id_uint ASC
        LIMIT {page_size:UInt32}
      )
      ORDER BY id_uint DESC
    `;
  }

  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
      query_params,
    });
    const rows = await resultSet.json<InferenceByIdRow>();
    return z.array(inferenceByIdRowSchema).parse(rows);
  } catch (error) {
    console.error(error);
    throw data("Error querying inference table", { status: 500 });
  }
}

export async function queryInferenceTableBounds(params?: {
  extraWhere?: string[];
  extraParams?: Record<string, string | number>;
}): Promise<TableBounds> {
  const { extraWhere = [], extraParams = {} } = params ?? {};

  // Build WHERE clause
  const whereClauses = [...extraWhere];
  const whereClause =
    whereClauses.length > 0 ? `WHERE ${whereClauses.join(" AND ")}` : "";

  const query = `
  SELECT
    (SELECT uint_to_uuid(id_uint) FROM InferenceById WHERE id_uint = (SELECT MIN(id_uint) FROM InferenceById ${whereClause})) AS first_id,
    (SELECT uint_to_uuid(id_uint) FROM InferenceById WHERE id_uint = (SELECT MAX(id_uint) FROM InferenceById ${whereClause})) AS last_id
  FROM InferenceById
  LIMIT 1
  `;

  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
      query_params: extraParams,
    });

    const rows = await resultSet.json<TableBounds>();
    if (!rows.length) {
      return {
        first_id: null,
        last_id: null,
      };
    }

    return TableBoundsSchema.parse(rows[0]);
  } catch (error) {
    console.error("Failed to query inference table bounds:", error);
    return {
      first_id: null,
      last_id: null,
    };
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
        uint_to_uuid(episode_id_uint) as episode_id,
        toUInt32(count(*)) as count,
        formatDateTime(min(UUIDv7ToDateTime(uint_to_uuid(id_uint))), '%Y-%m-%dT%H:%i:%SZ') as start_time,
        formatDateTime(max(UUIDv7ToDateTime(uint_to_uuid(id_uint))), '%Y-%m-%dT%H:%i:%SZ') as end_time,
        uint_to_uuid(max(id_uint)) as last_inference_id
      FROM InferenceByEpisodeId
      GROUP BY episode_id
      ORDER BY toUInt128(last_inference_id) DESC
      LIMIT {page_size:UInt32}
    `;
  } else if (before) {
    query = `
      SELECT
        uint_to_uuid(episode_id_uint) as episode_id,
        toUInt32(count(*)) as count,
        formatDateTime(min(UUIDv7ToDateTime(uint_to_uuid(id_uint))), '%Y-%m-%dT%H:%i:%SZ') as start_time,
        formatDateTime(max(UUIDv7ToDateTime(uint_to_uuid(id_uint))), '%Y-%m-%dT%H:%i:%SZ') as end_time,
        uint_to_uuid(max(id_uint)) as last_inference_id
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
          uint_to_uuid(episode_id_uint) as episode_id,
          toUInt32(count(*)) as count,
          formatDateTime(min(UUIDv7ToDateTime(uint_to_uuid(id_uint))), '%Y-%m-%dT%H:%i:%SZ') as start_time,
          formatDateTime(max(UUIDv7ToDateTime(uint_to_uuid(id_uint))), '%Y-%m-%dT%H:%i:%SZ') as end_time,
          uint_to_uuid(max(id_uint)) as last_inference_id,
          max(id_uint) as last_inference_id_uint
        FROM InferenceByEpisodeId
        GROUP BY episode_id
        HAVING last_inference_id_uint > toUInt128(toUUID({after:String}))
        ORDER BY last_inference_id_uint ASC
        LIMIT {page_size:UInt32}
      )
      ORDER BY last_inference_id_uint DESC
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
    return z.array(episodeByIdSchema).parse(rows);
  } catch (error) {
    console.error(error);
    throw data("Error querying episode table", { status: 500 });
  }
}

/// NOTE: these are the last inference IDs of the episodes that have the earliest and latest last inferences,
/// i.e. the first and last episodes in the table sort order
/// You should still paginate on the Inference IDs not the Episode IDs
export async function queryEpisodeTableBounds(): Promise<TableBounds> {
  const query = `
    SELECT
     (SELECT uint_to_uuid(id_uint) FROM InferenceByEpisodeId WHERE id_uint = (SELECT MIN(id_uint) FROM InferenceByEpisodeId)) AS first_id,
     (SELECT uint_to_uuid(id_uint) FROM InferenceByEpisodeId WHERE id_uint = (SELECT MAX(id_uint) FROM InferenceByEpisodeId)) AS last_id
    FROM InferenceByEpisodeId
    LIMIT 1
  `;
  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
    });
    const rows = await resultSet.json<TableBounds>();
    if (!rows.length) {
      return {
        first_id: null,
        last_id: null,
      };
    }
    return TableBoundsSchema.parse(rows[0]);
  } catch (error) {
    console.error(error);
    throw data("Error querying inference table bounds", { status: 500 });
  }
}

export async function queryInferenceTableByEpisodeId(params: {
  episode_id: string;
  page_size: number;
  before?: string;
  after?: string;
}): Promise<InferenceByIdRow[]> {
  return queryInferenceTable({
    page_size: params.page_size,
    before: params.before,
    after: params.after,
    extraWhere: ["episode_id = {episode_id:String}"],
    extraParams: { episode_id: params.episode_id },
  });
}

export async function queryInferenceTableBoundsByEpisodeId(params: {
  episode_id: string;
}): Promise<TableBounds> {
  return queryInferenceTableBounds({
    extraWhere: ["episode_id = {episode_id:String}"],
    extraParams: { episode_id: params.episode_id },
  });
}

export async function queryInferenceTableByFunctionName(params: {
  function_name: string;
  page_size: number;
  before?: string;
  after?: string;
}): Promise<InferenceByIdRow[]> {
  return queryInferenceTable({
    page_size: params.page_size,
    before: params.before,
    after: params.after,
    extraWhere: ["function_name = {function_name:String}"],
    extraParams: { function_name: params.function_name },
  });
}

export async function queryInferenceTableBoundsByFunctionName(params: {
  function_name: string;
}): Promise<TableBounds> {
  return queryInferenceTableBounds({
    extraWhere: ["function_name = {function_name:String}"],
    extraParams: { function_name: params.function_name },
  });
}

export async function queryInferenceTableByVariantName(params: {
  function_name: string;
  variant_name: string;
  page_size: number;
  before?: string;
  after?: string;
}): Promise<InferenceByIdRow[]> {
  return queryInferenceTable({
    page_size: params.page_size,
    before: params.before,
    after: params.after,
    extraWhere: [
      "function_name = {function_name:String}",
      "variant_name = {variant_name:String}",
    ],
    extraParams: {
      function_name: params.function_name,
      variant_name: params.variant_name,
    },
  });
}

export async function queryInferenceTableBoundsByVariantName(params: {
  function_name: string;
  variant_name: string;
}): Promise<TableBounds> {
  return queryInferenceTableBounds({
    extraWhere: [
      "function_name = {function_name:String}",
      "variant_name = {variant_name:String}",
    ],
    extraParams: {
      function_name: params.function_name,
      variant_name: params.variant_name,
    },
  });
}

export async function countInferencesForFunction(
  function_name: string,
  function_config: FunctionConfig,
): Promise<number> {
  const inference_table_name = getInferenceTableName(function_config);
  const query = `SELECT toUInt32(COUNT()) AS count FROM ${inference_table_name} WHERE function_name = {function_name:String}`;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { function_name },
  });
  const rows = await resultSet.json<{ count: number }>();
  return rows[0].count;
}

export async function countInferencesForVariant(
  function_name: string,
  function_config: FunctionConfig,
  variant_name: string,
): Promise<number> {
  const inference_table_name = getInferenceTableName(function_config);
  const query = `SELECT toUInt32(COUNT()) AS count FROM ${inference_table_name} WHERE function_name = {function_name:String} AND variant_name = {variant_name:String}`;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { function_name, variant_name },
  });
  const rows = await resultSet.json<{ count: number }>();
  return rows[0].count;
}

export async function countInferencesForEpisode(
  episode_id: string,
): Promise<number> {
  const query = `SELECT COUNT() AS count FROM InferenceByEpisodeId WHERE episode_id_uint = toUInt128(toUUID({episode_id:String}))`;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { episode_id },
  });
  const rows = await resultSet.json<{ count: string }>();
  return Number(rows[0].count);
}

export const chatInferenceRowSchema = z.object({
  id: z.string().uuid(),
  function_name: z.string(),
  variant_name: z.string(),
  episode_id: z.string().uuid(),
  input: z.string(),
  output: z.string(),
  tool_params: z.string(),
  inference_params: z.string(),
  processing_time_ms: z.number(),
  timestamp: z.string().datetime(),
  tags: z.record(z.string(), z.string()).default({}),
});

export type ChatInferenceRow = z.infer<typeof chatInferenceRowSchema>;

export const jsonInferenceRowSchema = z.object({
  id: z.string().uuid(),
  function_name: z.string(),
  variant_name: z.string(),
  episode_id: z.string().uuid(),
  input: z.string(),
  output: z.string(),
  output_schema: z.string(),
  inference_params: z.string(),
  processing_time_ms: z.number(),
  timestamp: z.string().datetime(),
  tags: z.record(z.string(), z.string()).default({}),
});

export type JsonInferenceRow = z.infer<typeof jsonInferenceRowSchema>;

export const inferenceRowSchema = z.discriminatedUnion("function_type", [
  chatInferenceRowSchema.extend({
    function_type: z.literal("chat"),
  }),
  jsonInferenceRowSchema.extend({
    function_type: z.literal("json"),
  }),
]);

export type InferenceRow = z.infer<typeof inferenceRowSchema>;

export const parsedChatInferenceRowSchema = chatInferenceRowSchema
  .omit({
    input: true,
    output: true,
    inference_params: true,
    tool_params: true,
  })
  .extend({
    input: inputSchema,
    output: z.array(contentBlockOutputSchema),
    inference_params: z.record(z.string(), z.unknown()),
    tool_params: z.record(z.string(), z.unknown()),
  });

export type ParsedChatInferenceRow = z.infer<
  typeof parsedChatInferenceRowSchema
>;

export const parsedJsonInferenceRowSchema = jsonInferenceRowSchema
  .omit({
    input: true,
    output: true,
    inference_params: true,
    output_schema: true,
  })
  .extend({
    input: inputSchema,
    output: jsonInferenceOutputSchema,
    inference_params: z.record(z.string(), z.unknown()),
    output_schema: z.record(z.string(), z.unknown()),
  });

export type ParsedJsonInferenceRow = z.infer<
  typeof parsedJsonInferenceRowSchema
>;

export const parsedInferenceRowSchema = z.discriminatedUnion("function_type", [
  parsedChatInferenceRowSchema.extend({
    function_type: z.literal("chat"),
  }),
  parsedJsonInferenceRowSchema.extend({
    function_type: z.literal("json"),
  }),
]);

export type ParsedInferenceRow = z.infer<typeof parsedInferenceRowSchema>;

export function parseInferenceOutput(
  output: string,
): ContentBlockOutput[] | JsonInferenceOutput {
  const parsed = JSON.parse(output);
  if (Array.isArray(parsed)) {
    return z.array(contentBlockOutputSchema).parse(parsed);
  }
  return jsonInferenceOutputSchema.parse(parsed);
}

function parseInferenceRow(row: InferenceRow): ParsedInferenceRow {
  if (row.function_type === "chat") {
    return {
      ...row,
      input: inputSchema.parse(JSON.parse(row.input)),
      output: parseInferenceOutput(row.output) as ContentBlockOutput[],
      inference_params: z
        .record(z.string(), z.unknown())
        .parse(JSON.parse(row.inference_params)),
      tool_params:
        row.tool_params === ""
          ? {}
          : z
              .record(z.string(), z.unknown())
              .parse(JSON.parse(row.tool_params)),
    };
  } else {
    return {
      ...row,
      input: inputSchema.parse(JSON.parse(row.input)),
      output: parseInferenceOutput(row.output) as JsonInferenceOutput,
      inference_params: z
        .record(z.string(), z.unknown())
        .parse(JSON.parse(row.inference_params)),
      output_schema: z
        .record(z.string(), z.unknown())
        .parse(JSON.parse(row.output_schema)),
    };
  }
}

export async function queryInferenceById(
  id: string,
): Promise<ParsedInferenceRow | null> {
  const query = `
    SELECT
  uint_to_uuid(i.id_uint) AS id,

  -- Common columns (pick via IF)
  IF(i.function_type = 'chat', c.function_name, j.function_name) AS function_name,
  IF(i.function_type = 'chat', c.variant_name,   j.variant_name)   AS variant_name,
  IF(i.function_type = 'chat', c.episode_id,     j.episode_id)     AS episode_id,
  IF(i.function_type = 'chat', c.input,          j.input)          AS input,
  IF(i.function_type = 'chat', c.output,         j.output)         AS output,

  -- Chat-specific columns
  IF(i.function_type = 'chat', c.tool_params, '') AS tool_params,

  -- Inference params (common name in the union)
  IF(i.function_type = 'chat', c.inference_params, j.inference_params) AS inference_params,

  -- Processing time
  IF(i.function_type = 'chat', c.processing_time_ms, j.processing_time_ms) AS processing_time_ms,

  -- JSON-specific column
  IF(i.function_type = 'json', j.output_schema, '') AS output_schema,

  -- Timestamps & tags
  IF(i.function_type = 'chat',
   formatDateTime(c.timestamp, '%Y-%m-%dT%H:%i:%SZ'),
   formatDateTime(j.timestamp, '%Y-%m-%dT%H:%i:%SZ')
) AS timestamp,
  IF(i.function_type = 'chat', c.tags,      j.tags)      AS tags,

  -- Discriminator itself
  i.function_type

FROM InferenceById i
LEFT JOIN ChatInference c
  ON i.id_uint = toUInt128(c.id)
LEFT JOIN JsonInference j
  ON i.id_uint = toUInt128(j.id)
WHERE uint_to_uuid(i.id_uint) = {id:String};
  `;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { id },
  });
  const rows = await resultSet.json<InferenceRow>();
  const firstRow = rows[0];
  if (!firstRow) return null;
  const parsedRow = parseInferenceRow(firstRow);
  return parsedRow;
}

export const modelInferenceRowSchema = z.object({
  id: z.string().uuid(),
  inference_id: z.string().uuid(),
  raw_request: z.string(),
  raw_response: z.string(),
  model_name: z.string(),
  model_provider_name: z.string(),
  input_tokens: z.number().nullable(),
  output_tokens: z.number().nullable(),
  response_time_ms: z.number(),
  ttft_ms: z.number().nullable(),
  timestamp: z.string().datetime(),
  system: z.string().nullable(),
  input_messages: z.string(),
  output: z.string(),
});

export type ModelInferenceRow = z.infer<typeof modelInferenceRowSchema>;

export const parsedModelInferenceRowSchema = modelInferenceRowSchema
  .omit({
    input_messages: true,
    output: true,
  })
  .extend({
    input_messages: z.array(requestMessageSchema),
    output: z.array(contentBlockSchema),
  });

export type ParsedModelInferenceRow = z.infer<
  typeof parsedModelInferenceRowSchema
>;

function parseModelInferenceRow(
  row: ModelInferenceRow,
): ParsedModelInferenceRow {
  return {
    ...row,
    input_messages: z
      .array(requestMessageSchema)
      .parse(JSON.parse(row.input_messages)),
    output: z.array(contentBlockSchema).parse(JSON.parse(row.output)),
  };
}

export async function queryModelInferencesByInferenceId(
  id: string,
): Promise<ParsedModelInferenceRow[]> {
  const query = `
    SELECT *, formatDateTime(timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp FROM ModelInference WHERE inference_id = {id:String}
  `;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { id },
  });
  const rows = await resultSet.json<ModelInferenceRow>();
  const validatedRows = z.array(modelInferenceRowSchema).parse(rows);
  const parsedRows = validatedRows.map(parseModelInferenceRow);
  return parsedRows;
}

const functionCountInfoSchema = z.object({
  function_name: z.string(),
  max_timestamp: z.string().datetime(),
  count: z.number(),
});

export type FunctionCountInfo = z.infer<typeof functionCountInfoSchema>;

export async function countInferencesByFunction(): Promise<
  FunctionCountInfo[]
> {
  const query = `SELECT
        function_name,
        formatDateTime(max(timestamp), '%Y-%m-%dT%H:%i:%SZ') AS max_timestamp,
        toUInt32(count()) AS count
    FROM (
        SELECT function_name, timestamp
        FROM ChatInference
        UNION ALL
        SELECT function_name, timestamp
        FROM JsonInference
    )
    GROUP BY function_name
    ORDER BY max_timestamp DESC`;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
  });
  const rows = await resultSet.json<FunctionCountInfo[]>();
  const validatedRows = z.array(functionCountInfoSchema).parse(rows);
  return validatedRows;
}

export async function countEpisodes(): Promise<number> {
  const query = `SELECT COUNT(DISTINCT episode_id) as count FROM (
    SELECT episode_id FROM ChatInference
    UNION ALL
    SELECT episode_id FROM JsonInference
  )`;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
  });
  const rows = await resultSet.json<{ count: number }>();
  return rows[0].count;
}
