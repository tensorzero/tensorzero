import {
  modelInferenceInputMessageSchema,
  type TableBounds,
  type TableBoundsWithCount,
  JsonValueSchema,
} from "./common";
import { contentBlockOutputSchema, inputSchema } from "./common";
import { data } from "react-router";
import type {
  FunctionConfig,
  JsonInferenceOutput,
  ContentBlockChatOutput,
  InferenceByIdRow,
  InferenceRow,
  ModelInferenceRow,
  AdjacentIds as NativeAdjacentIds,
  TableBoundsWithCount as NativeTableBoundsWithCount,
  InferenceTableFilter,
  QueryInferenceTableParams,
  QueryInferenceTableBoundsParams,
  ToolCallConfigDatabaseInsert,
} from "tensorzero-node";
import { getNativeDatabaseClient } from "../tensorzero/native_client.server";
import { resolveInput, resolveModelInferenceMessages } from "../resolve.server";
import {
  parseInferenceOutput,
  type AdjacentIds,
  type InferenceExtraBody,
  type ParsedInferenceRow,
  type ParsedModelInferenceRow,
} from "./inference";
import { z } from "zod";
import { logger } from "~/utils/logger";
import { getConfig, getFunctionConfig } from "../config/index.server";

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
 *
 * TODO (#2788): Create MVs for sorting episodes and inferences by ID DESC
 */
export async function queryInferenceTable(params: {
  page_size: number;
  before?: string;
  after?: string;
  filter?: InferenceTableFilter;
}): Promise<InferenceByIdRow[]> {
  const { page_size, before, after, filter } = params;

  if (before && after) {
    throw new Error("Cannot specify both 'before' and 'after' parameters");
  }

  try {
    const databaseClient = await getNativeDatabaseClient();
    const queryParams: QueryInferenceTableParams = {
      page_size,
      before,
      after,
      filter,
    };
    return await databaseClient.queryInferenceTable(queryParams);
  } catch (error) {
    logger.error(error);
    throw data("Error querying inference table", { status: 500 });
  }
}

/// TODO (#2788): Create MVs for sorting episodes and inferences by ID DESC
export async function queryInferenceTableBounds(params?: {
  filter?: InferenceTableFilter;
}): Promise<TableBoundsWithCount> {
  try {
    const databaseClient = await getNativeDatabaseClient();
    const queryParams: QueryInferenceTableBoundsParams = {
      filter: params?.filter,
    };
    const bounds = (await databaseClient.queryInferenceTableBounds(
      queryParams,
    )) as NativeTableBoundsWithCount;
    return {
      first_id: bounds.first_id,
      last_id: bounds.last_id,
      count: Number(bounds.count),
    } satisfies TableBoundsWithCount;
  } catch (error) {
    logger.error("Failed to query inference table bounds:", error);
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
    filter: { episode_id: params.episode_id },
  });
}

export async function queryInferenceTableBoundsByEpisodeId(params: {
  episode_id: string;
}): Promise<TableBounds> {
  const bounds = await queryInferenceTableBounds({
    filter: { episode_id: params.episode_id },
  });
  return {
    first_id: bounds.first_id,
    last_id: bounds.last_id,
  };
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
    filter: { function_name: params.function_name },
  });
}

export async function queryInferenceTableBoundsByFunctionName(params: {
  function_name: string;
}): Promise<TableBounds> {
  const bounds = await queryInferenceTableBounds({
    filter: { function_name: params.function_name },
  });
  return {
    first_id: bounds.first_id,
    last_id: bounds.last_id,
  };
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
    filter: {
      function_name: params.function_name,
      variant_name: params.variant_name,
    },
  });
}

export async function queryInferenceTableBoundsByVariantName(params: {
  function_name: string;
  variant_name: string;
}): Promise<TableBounds> {
  const bounds = await queryInferenceTableBounds({
    filter: {
      function_name: params.function_name,
      variant_name: params.variant_name,
    },
  });
  return {
    first_id: bounds.first_id,
    last_id: bounds.last_id,
  };
}

export async function countInferencesForFunction(
  function_name: string,
  function_config: FunctionConfig,
): Promise<number> {
  const databaseClient = await getNativeDatabaseClient();
  return databaseClient.countInferencesForFunction(
    function_name,
    function_config.type,
  );
}

export async function countInferencesForVariant(
  function_name: string,
  function_config: FunctionConfig,
  variant_name: string,
): Promise<number> {
  const databaseClient = await getNativeDatabaseClient();
  return databaseClient.countInferencesForVariant(
    function_name,
    function_config.type,
    variant_name,
  );
}

export async function countInferencesForEpisode(
  episode_id: string,
): Promise<number> {
  const databaseClient = await getNativeDatabaseClient();
  return databaseClient.countInferencesForEpisode(episode_id);
}

function normalizeTags(tags: InferenceRow["tags"]): Record<string, string> {
  return Object.fromEntries(
    Object.entries(tags ?? {}).filter(
      (entry): entry is [string, string] => entry[1] != null,
    ),
  );
}

function normalizeAdjacentIds(adjacentIds: NativeAdjacentIds): AdjacentIds {
  return {
    previous_id:
      adjacentIds.previous_id === undefined || adjacentIds.previous_id === null
        ? null
        : adjacentIds.previous_id,
    next_id:
      adjacentIds.next_id === undefined || adjacentIds.next_id === null
        ? null
        : adjacentIds.next_id,
  };
}

async function parseInferenceRow(
  row: InferenceRow,
): Promise<ParsedInferenceRow> {
  const input = inputSchema.parse(JSON.parse(row.input));
  const config = await getConfig();
  const functionConfig = await getFunctionConfig(row.function_name, config);
  const resolvedInput = await resolveInput(input, functionConfig);
  const extra_body = row.extra_body
    ? (JSON.parse(row.extra_body) as InferenceExtraBody[])
    : null;
  const normalizedRow = {
    ...row,
    processing_time_ms: Number(row.processing_time_ms),
    tags: normalizeTags(row.tags),
  };
  if (row.function_type === "chat") {
    const tool_params: ToolCallConfigDatabaseInsert | null =
      row.tool_params === null || row.tool_params === ""
        ? null
        : (JSON.parse(row.tool_params) as ToolCallConfigDatabaseInsert);
    const { output_schema: chatOutputSchema, ...chatRowBase } = normalizedRow;
    void chatOutputSchema;
    return {
      ...chatRowBase,
      function_type: "chat" as const,
      input: resolvedInput,
      output: parseInferenceOutput(row.output) as ContentBlockChatOutput[],
      inference_params: JSON.parse(row.inference_params) as Record<
        string,
        unknown
      >,
      tool_params: tool_params,
      extra_body,
    };
  } else {
    const { tool_params: jsonToolParams, ...jsonRowBase } = normalizedRow;
    void jsonToolParams;
    return {
      ...jsonRowBase,
      function_type: "json" as const,
      input: resolvedInput,
      output: parseInferenceOutput(row.output) as JsonInferenceOutput,
      inference_params: JSON.parse(row.inference_params) as Record<
        string,
        unknown
      >,
      output_schema: JsonValueSchema.parse(
        JSON.parse(row.output_schema ?? "null"),
      ),
      extra_body,
    };
  }
}

export async function queryInferenceById(
  id: string,
): Promise<ParsedInferenceRow | null> {
  const databaseClient = await getNativeDatabaseClient();
  const row = await databaseClient.queryInferenceById(id);
  if (!row) {
    return null;
  }
  return parseInferenceRow(row);
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
    input_tokens: row.input_tokens === null ? null : Number(row.input_tokens),
    output_tokens:
      row.output_tokens === null ? null : Number(row.output_tokens),
    response_time_ms: Number(row.response_time_ms),
    ttft_ms: row.ttft_ms === null ? null : Number(row.ttft_ms),
    tags: normalizeTags(row.tags),
  };
  return processedRow;
}

export async function queryModelInferencesByInferenceId(
  id: string,
): Promise<ParsedModelInferenceRow[]> {
  const databaseClient = await getNativeDatabaseClient();
  const rows = await databaseClient.queryModelInferencesByInferenceId(id);
  const parsedRows = await Promise.all(rows.map(parseModelInferenceRow));
  return parsedRows;
}

export type FunctionCountInfo = {
  function_name: string;
  max_timestamp: string;
  count: number;
};

export async function countInferencesByFunction(): Promise<
  FunctionCountInfo[]
> {
  const databaseClient = await getNativeDatabaseClient();
  const rows = await databaseClient.countInferencesByFunction();
  return rows.map((row) => ({
    function_name: row.function_name,
    max_timestamp: row.max_timestamp,
    count: Number(row.count),
  }));
}

export async function getAdjacentInferenceIds(
  currentInferenceId: string,
): Promise<AdjacentIds> {
  const databaseClient = await getNativeDatabaseClient();
  const adjacentIds = (await databaseClient.getAdjacentInferenceIds(
    currentInferenceId,
  )) as NativeAdjacentIds;
  return normalizeAdjacentIds(adjacentIds);
}

export async function getAdjacentEpisodeIds(
  currentEpisodeId: string,
): Promise<AdjacentIds> {
  const databaseClient = await getNativeDatabaseClient();
  const adjacentIds = (await databaseClient.getAdjacentEpisodeIds(
    currentEpisodeId,
  )) as NativeAdjacentIds;
  return normalizeAdjacentIds(adjacentIds);
}
