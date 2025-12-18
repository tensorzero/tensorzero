import { data } from "react-router";
import type { StoredInference, InferenceFilter } from "~/types/tensorzero";
import { getClickhouseClient } from "./client.server";
import { z } from "zod";
import { logger } from "~/utils/logger";
import { getTensorZeroClient } from "../tensorzero.server";
import { isTensorZeroServerError } from "../tensorzero";
import { applyPaginationLogic } from "../pagination";

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
  filters?: InferenceFilter;
  search_query?: string;
}): Promise<ListInferencesResult> {
  const {
    limit,
    before,
    after,
    function_name,
    variant_name,
    episode_id,
    filters,
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
      filters,
      search_query_experimental: search_query,
    });

    const {
      items: inferences,
      hasNextPage,
      hasPreviousPage,
    } = applyPaginationLogic(response.inferences, limit, { before, after });

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
  const result = await client.getInferenceStats(function_name, {
    variantName: variant_name,
  });
  return Number(result.inference_count);
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
