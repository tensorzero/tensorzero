import {
  CountSchema,
  modelInferenceInputMessageSchema,
  contentBlockOutputSchema,
} from "./common";
import { data } from "react-router";
import type { StoredInference, InferenceFilter } from "~/types/tensorzero";
import { getClickhouseClient } from "./client.server";
import { resolveModelInferenceMessages } from "../resolve.server";
import {
  modelInferenceRowSchema,
  parsedModelInferenceRowSchema,
  type ModelInferenceRow,
  type ParsedModelInferenceRow,
} from "./inference";
import { z } from "zod";
import { logger } from "~/utils/logger";
import { getTensorZeroClient } from "../tensorzero.server";
import { isTensorZeroServerError } from "../tensorzero";

/**
 * Result type for listInferencesWithPagination with pagination info.
 */
export type ListInferencesResult = {
  inferences: StoredInference[];
  hasNextPage: boolean;
  hasPreviousPage: boolean;
};

/**
 * Lists inferences using the public v1 API with cursor-based pagination.
 * Implements the limit+1 pattern to detect if there are more pages.
 *
 * @param params - Query parameters for listing inferences
 * @returns Inferences with pagination info (hasNextPage, hasPreviousPage)
 */
export async function listInferencesWithPagination(params: {
  limit: number;
  before?: string; // UUIDv7 string - get inferences before this ID (going to older)
  after?: string; // UUIDv7 string - get inferences after this ID (going to newer)
  function_name?: string;
  variant_name?: string;
  episode_id?: string;
  filter?: InferenceFilter;
  search_query?: string;
}): Promise<ListInferencesResult> {
  const {
    limit,
    before,
    after,
    function_name,
    variant_name,
    episode_id,
    filter,
    search_query,
  } = params;

  if (before && after) {
    throw new Error("Cannot specify both 'before' and 'after' parameters");
  }

  try {
    const client = getTensorZeroClient();

    // Request limit + 1 to detect if there are more pages
    const response = await client.listInferences({
      output_source: "inference",
      limit: limit + 1,
      before,
      after,
      function_name,
      variant_name,
      episode_id,
      filter,
      search_query_experimental: search_query,
    });

    // Determine if there are more pages based on whether we got more than limit results
    const hasMore = response.inferences.length > limit;

    // Only return up to limit inferences (hide the extra one used for detection)
    // For 'after' pagination, the extra item is at the BEGINNING after the backend reverses
    // For 'before' pagination (or no pagination), the extra item is at the END
    let inferences: typeof response.inferences;
    if (hasMore) {
      if (after) {
        // Extra item is at position 0, so take items from position 1 onwards
        inferences = response.inferences.slice(1, limit + 1);
      } else {
        // Extra item is at the end, so take first 'limit' items
        inferences = response.inferences.slice(0, limit);
      }
    } else {
      inferences = response.inferences;
    }

    // Pagination direction logic:
    // - When using 'before': we're going to older inferences (next page = older)
    // - When using 'after': we're going to newer inferences (previous page = newer)
    // - When neither: we're on the first page (most recent)
    let hasNextPage: boolean;
    let hasPreviousPage: boolean;

    if (before) {
      // Going backwards in time (older). hasMore means there are older pages.
      hasNextPage = hasMore;
      // We came from a newer page, so there's always a previous (newer) page
      hasPreviousPage = true;
    } else if (after) {
      // Going forwards in time (newer). hasMore means there are newer pages.
      // But since results are displayed newest first, "previous" button goes to newer
      hasPreviousPage = hasMore;
      // We came from an older page, so there's always a next (older) page
      hasNextPage = true;
    } else {
      // Initial page load - showing most recent
      hasNextPage = hasMore;
      hasPreviousPage = false;
    }

    return {
      inferences,
      hasNextPage,
      hasPreviousPage,
    };
  } catch (error) {
    logger.error("Failed to list inferences:", error);
    if (isTensorZeroServerError(error)) {
      throw data("Error listing inferences", {
        status: error.status,
        statusText: error.statusText ?? undefined,
      });
    }
    throw data("Error listing inferences", { status: 500 });
  }
}

export async function countInferencesForFunction(
  function_name: string,
): Promise<number> {
  const client = getTensorZeroClient();
  const result = await client.getInferenceStats(function_name);
  return Number(result.inference_count);
}

export async function countInferencesForVariant(
  function_name: string,
  variant_name: string,
): Promise<number> {
  const client = getTensorZeroClient();
  const result = await client.getInferenceStats(function_name, variant_name);
  return Number(result.inference_count);
}

export async function countInferencesForEpisode(
  episode_id: string,
): Promise<number> {
  const query = `SELECT toUInt32(COUNT()) AS count FROM InferenceByEpisodeId FINAL WHERE episode_id_uint = toUInt128(toUUID({episode_id:String}))`;
  const resultSet = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: { episode_id },
  });
  const rows = await resultSet.json<{ count: string }>();
  const parsedRows = rows.map((row) => CountSchema.parse(row));
  return parsedRows[0].count;
}

async function parseModelInferenceRow(
  row: ModelInferenceRow,
): Promise<ParsedModelInferenceRow> {
  const parsedMessages = z
    .array(modelInferenceInputMessageSchema)
    .parse(JSON.parse(row.input_messages));
  const resolvedMessages = await resolveModelInferenceMessages(parsedMessages);
  const processedRow = {
    ...row,
    input_messages: resolvedMessages,
    output: z.array(contentBlockOutputSchema).parse(JSON.parse(row.output)),
  };
  return parsedModelInferenceRowSchema.parse(processedRow);
}

export async function queryModelInferencesByInferenceId(
  id: string,
): Promise<ParsedModelInferenceRow[]> {
  const query = `
    SELECT *, formatDateTime(timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp FROM ModelInference WHERE inference_id = {id:String}
  `;
  const resultSet = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
    query_params: { id },
  });
  const rows = await resultSet.json<ModelInferenceRow>();
  const validatedRows = z.array(modelInferenceRowSchema).parse(rows);
  const parsedRows = await Promise.all(
    validatedRows.map(parseModelInferenceRow),
  );
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
  const resultSet = await getClickhouseClient().query({
    query,
    format: "JSONEachRow",
  });
  const rows = await resultSet.json<FunctionCountInfo[]>();
  const validatedRows = z.array(functionCountInfoSchema).parse(rows);
  return validatedRows;
}
