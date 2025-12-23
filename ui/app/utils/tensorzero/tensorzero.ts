/*
TensorZero Client (for internal use only for now)

TODO(shuyangli): Figure out a way to generate the HTTP client, possibly from Schema.
*/

import { z } from "zod";
import {
  contentBlockChatOutputSchema,
  thoughtContentSchema,
  ZodJsonValueSchema,
  type ZodStoragePath,
} from "~/utils/clickhouse/common";
import { GatewayConnectionError, TensorZeroServerError } from "./errors";
import type {
  CloneDatapointsResponse,
  CountFeedbackByTargetIdResponse,
  CountInferencesRequest,
  CountInferencesResponse,
  CountModelsResponse,
  CountWorkflowEvaluationRunEpisodesByTaskNameResponse,
  CountWorkflowEvaluationRunEpisodesResponse,
  CountWorkflowEvaluationRunsResponse,
  CreateDatapointsFromInferenceRequest,
  CumulativeFeedbackTimeSeriesPoint,
  DatapointStatsResponse,
  DemonstrationFeedbackRow,
  EvaluationRunStatsResponse,
  CreateDatapointsRequest,
  CreateDatapointsResponse,
  FeedbackRow,
  MetricsWithFeedbackResponse,
  Datapoint,
  GetDatapointCountResponse,
  DeleteDatapointsRequest,
  DeleteDatapointsResponse,
  GetDemonstrationFeedbackResponse,
  GetFeedbackBoundsResponse,
  GetFeedbackByTargetIdResponse,
  GetFunctionThroughputByVariantResponse,
  GetModelLatencyResponse,
  GetModelUsageResponse,
  GetWorkflowEvaluationProjectCountResponse,
  GetWorkflowEvaluationProjectsResponse,
  GetWorkflowEvaluationRunEpisodesWithFeedbackResponse,
  GetWorkflowEvaluationRunsResponse,
  GetWorkflowEvaluationRunStatisticsResponse,
  InferenceWithFeedbackStatsResponse,
  GetDatapointsRequest,
  GetDatapointsResponse,
  GetInferencesRequest,
  GetInferencesResponse,
  GetModelInferencesResponse,
  InferenceStatsResponse,
  LatestFeedbackIdByMetricResponse,
  ListDatapointsRequest,
  ListDatasetsResponse,
  ListEvaluationRunsResponse,
  ListInferencesRequest,
  ListInferenceMetadataResponse,
  ListWorkflowEvaluationRunEpisodesByTaskNameResponse,
  ListWorkflowEvaluationRunsResponse,
  SearchEvaluationRunsResponse,
  SearchWorkflowEvaluationRunsResponse,
  StatusResponse,
  TimeWindow,
  TableBoundsWithCount,
  UiConfig,
  UpdateDatapointRequest,
  UpdateDatapointsMetadataRequest,
  UpdateDatapointsRequest,
  UpdateDatapointsResponse,
  ListEpisodesResponse,
  GetEpisodeInferenceCountResponse,
  GetEvaluationResultsResponse,
  GetEvaluationRunInfosResponse,
  GetEvaluationStatisticsResponse,
  VariantPerformancesResponse,
  InferenceStatsByVariant,
} from "~/types/tensorzero";

/**
 * Roles for input messages.
 */
export const RoleSchema = z.enum(["system", "user", "assistant", "tool"]);
export type Role = z.infer<typeof RoleSchema>;

/**
 * A tool call request.
 */
export const ToolCallSchema = z.object({
  name: z.string(),
  /** The arguments as a JSON string. */
  arguments: z.string(),
  id: z.string(),
});
export type ToolCall = z.infer<typeof ToolCallSchema>;

/**
 * A tool call result.
 */
export const ToolResultSchema = z.object({
  name: z.string(),
  result: z.string(),
  id: z.string(),
});
export type ToolResult = z.infer<typeof ToolResultSchema>;

export const TextContentSchema = z.object({
  type: z.literal("text"),
  text: z.string(),
});

export const TextArgumentsContentSchema = z.object({
  type: z.literal("text"),
  arguments: ZodJsonValueSchema,
});

export const TemplateContentSchema = z.object({
  type: z.literal("template"),
  name: z.string(),
  arguments: ZodJsonValueSchema,
});

export const RawTextContentSchema = z.object({
  type: z.literal("raw_text"),
  value: z.string(),
});

export const ToolCallContentSchema = z.object({
  type: z.literal("tool_call"),
  ...ToolCallSchema.shape,
});

export const ToolResultContentSchema = z.object({
  type: z.literal("tool_result"),
  ...ToolResultSchema.shape,
});

export const ImageContentSchema = z
  .object({
    type: z.literal("image"),
  })
  .and(
    z.union([
      z.object({
        url: z.string(),
      }),
      z.object({
        mime_type: z.string(),
        data: z.string(),
      }),
    ]),
  );
export type ImageContent = z.infer<typeof ImageContentSchema>;

/**
 * Unknown content type for model-specific content
 */
// TODO(shuyangli): There's a lot of duplication between this and ui/app/utils/clickhouse/common.ts. We should get rid of all of them and use Rust-generated bindings.
export const UnknownContentSchema = z.object({
  type: z.literal("unknown"),
  data: ZodJsonValueSchema,
  model_provider_name: z.string().nullish(),
});
export type UnknownContent = z.infer<typeof UnknownContentSchema>;

export const InputMessageContentSchema = z.union([
  TextContentSchema,
  TextArgumentsContentSchema,
  RawTextContentSchema,
  ToolCallContentSchema,
  ToolResultContentSchema,
  ImageContentSchema,
  thoughtContentSchema,
  UnknownContentSchema,
  TemplateContentSchema,
]);

export type InputMessageContent = z.infer<typeof InputMessageContentSchema>;

/**
 * An input message sent by the client.
 */
export const InputMessageSchema = z.object({
  role: RoleSchema,
  content: z.array(InputMessageContentSchema),
});
export type InputMessage = z.infer<typeof InputMessageSchema>;

/**
 * The inference input object.
 */
export const InputSchema = z.object({
  system: ZodJsonValueSchema.optional(),
  messages: z.array(InputMessageSchema),
});
export type Input = z.infer<typeof InputSchema>;

/**
 * A Tool that the LLM may call.
 */
export const ToolSchema = z.object({
  description: z.string(),
  parameters: ZodJsonValueSchema,
  name: z.string(),
  strict: z.boolean().optional(),
});
export type Tool = z.infer<typeof ToolSchema>;

/**
 * Tool choice, which controls how tools are selected.
 * This mirrors the Rust enum:
 * - "none": no tool should be used
 * - "auto": let the model decide
 * - "required": the model must call a tool
 * - { specific: "tool_name" }: force a specific tool
 */
export const ToolChoiceSchema = z.union([
  z.enum(["none", "auto", "required"]),
  z.object({ specific: z.string() }),
]);
export type ToolChoice = z.infer<typeof ToolChoiceSchema>;

/**
 * Inference parameters allow runtime overrides for a given variant.
 */
export const InferenceParamsSchema = z.record(z.record(ZodJsonValueSchema));
export type InferenceParams = z.infer<typeof InferenceParamsSchema>;

/**
 * The request type for inference. These fields correspond roughly
 * to the Rust `Params` struct.
 *
 * Exactly one of `function_name` or `model_name` should be provided.
 */
export const InferenceRequestSchema = z.object({
  function_name: z.string().optional(),
  model_name: z.string().optional(),
  episode_id: z.string().optional(),
  input: InputSchema,
  stream: z.boolean().optional(),
  params: InferenceParamsSchema.optional(),
  variant_name: z.string().optional(),
  dryrun: z.boolean().optional(),
  tags: z.record(z.string()).optional(),
  allowed_tools: z.array(z.string()).optional(),
  additional_tools: z.array(ToolSchema).optional(),
  tool_choice: ToolChoiceSchema.optional(),
  parallel_tool_calls: z.boolean().optional(),
  output_schema: ZodJsonValueSchema.optional(),
  credentials: z.record(z.string()).optional(),
});
export type InferenceRequest = z.infer<typeof InferenceRequestSchema>;

/**
 * Inference responses vary based on the function type.
 */
export const ChatInferenceResponseSchema = z.object({
  inference_id: z.string(),
  episode_id: z.string(),
  variant_name: z.string(),
  content: z.array(contentBlockChatOutputSchema),
  usage: z
    .object({
      input_tokens: z.number(),
      output_tokens: z.number(),
    })
    .optional(),
});
export type ChatInferenceResponse = z.infer<typeof ChatInferenceResponseSchema>;

export const JSONInferenceResponseSchema = z.object({
  inference_id: z.string(),
  episode_id: z.string(),
  variant_name: z.string(),
  output: z.object({
    raw: z.string(),
    parsed: ZodJsonValueSchema.nullable(),
  }),
  usage: z
    .object({
      input_tokens: z.number(),
      output_tokens: z.number(),
    })
    .optional(),
});
export type JSONInferenceResponse = z.infer<typeof JSONInferenceResponseSchema>;

/**
 * The overall inference response is a union of chat and JSON responses.
 */
export const InferenceResponseSchema = z.union([
  ChatInferenceResponseSchema,
  JSONInferenceResponseSchema,
]);
export type InferenceResponse = z.infer<typeof InferenceResponseSchema>;

/**
 * Feedback requests attach a metric value to a given inference or episode.
 */
export const FeedbackRequestSchema = z.object({
  dryrun: z.boolean().optional(),
  episode_id: z.string().nullable(),
  inference_id: z.string().nullable(),
  metric_name: z.string(),
  tags: z.record(z.string()).optional(),
  value: ZodJsonValueSchema,
  internal: z.boolean().optional(),
});
export type FeedbackRequest = z.infer<typeof FeedbackRequestSchema>;

export const FeedbackResponseSchema = z.object({
  feedback_id: z.string(),
});
export type FeedbackResponse = z.infer<typeof FeedbackResponseSchema>;

/**
 * Schema for datapoint response
 */
export const DatapointResponseSchema = z.object({
  id: z.string(),
});
export type DatapointResponse = z.infer<typeof DatapointResponseSchema>;

/**
 * Response type for getCumulativeFeedbackTimeseries endpoint
 */
export interface GetCumulativeFeedbackTimeseriesResponse {
  timeseries: CumulativeFeedbackTimeSeriesPoint[];
}

/**
 * A client for calling the TensorZero Gateway inference and feedback endpoints.
 */
export class TensorZeroClient {
  private baseUrl: string;
  private apiKey: string | null;

  /**
   * @param baseUrl - The base URL of the TensorZero Gateway (e.g. "http://localhost:3000")
   * @param apiKey - Optional API key for bearer authentication
   */
  constructor(baseUrl: string, apiKey?: string | null) {
    // Remove any trailing slash for consistency.
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.apiKey = apiKey ?? null;
  }

  /**
   * Sends feedback for a particular inference or episode.
   * @param request - The feedback request payload.
   * @returns A promise that resolves with the feedback response.
   */
  async feedback(request: FeedbackRequest): Promise<FeedbackResponse> {
    const response = await this.fetch("/feedback", {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as FeedbackResponse;
  }

  /**
   * Queries feedback for a given target ID with pagination support.
   * @param targetId - The target ID (inference_id or episode_id) to query feedback for
   * @param options - Optional pagination parameters
   * @returns A promise that resolves with a list of feedback rows
   * @throws Error if the request fails
   */
  async getFeedbackByTargetId(
    targetId: string,
    options?: {
      before?: string;
      after?: string;
      limit?: number;
    },
  ): Promise<FeedbackRow[]> {
    const params = new URLSearchParams();
    if (options?.before) params.set("before", options.before);
    if (options?.after) params.set("after", options.after);
    if (options?.limit) params.set("limit", options.limit.toString());

    const queryString = params.toString();
    const endpoint = `/internal/feedback/${encodeURIComponent(targetId)}${queryString ? `?${queryString}` : ""}`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body = (await response.json()) as GetFeedbackByTargetIdResponse;
    return body.feedback;
  }

  /**
   * Gets demonstration feedback for a given inference.
   * @param inferenceId - The inference ID to get demonstration feedback for
   * @param options - Optional pagination parameters
   * @returns A promise that resolves with a list of demonstration feedback rows
   * @throws Error if the request fails
   */
  async getDemonstrationFeedback(
    inferenceId: string,
    options?: {
      before?: string;
      after?: string;
      limit?: number;
    },
  ): Promise<DemonstrationFeedbackRow[]> {
    const params = new URLSearchParams();
    if (options?.before) params.set("before", options.before);
    if (options?.after) params.set("after", options.after);
    if (options?.limit) params.set("limit", options.limit.toString());

    const queryString = params.toString();
    const endpoint = `/internal/feedback/${encodeURIComponent(inferenceId)}/demonstrations${queryString ? `?${queryString}` : ""}`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body = (await response.json()) as GetDemonstrationFeedbackResponse;
    return body.feedback;
  }

  /**
   * Gets model usage timeseries data.
   * @param timeWindow The time window granularity for grouping data
   * @param maxPeriods Maximum number of time periods to return
   * @returns A promise that resolves with the model usage timeseries data
   * @throws Error if the request fails
   */
  async getModelUsageTimeseries(
    timeWindow: TimeWindow,
    maxPeriods: number,
  ): Promise<GetModelUsageResponse> {
    const params = new URLSearchParams({
      time_window: timeWindow,
      max_periods: maxPeriods.toString(),
    });
    const response = await this.fetch(
      `/internal/models/usage?${params.toString()}`,
      {
        method: "GET",
      },
    );
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetModelUsageResponse;
  }

  /**
   * Gets model latency quantile distributions.
   * @param timeWindow The time window for aggregating latency data
   * @returns A promise that resolves with the model latency quantiles
   * @throws Error if the request fails
   */
  async getModelLatencyQuantiles(
    timeWindow: TimeWindow,
  ): Promise<GetModelLatencyResponse> {
    const params = new URLSearchParams({
      time_window: timeWindow,
    });
    const response = await this.fetch(
      `/internal/models/latency?${params.toString()}`,
      {
        method: "GET",
      },
    );
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetModelLatencyResponse;
  }

  /**
   * Inserts a datapoint from an existing inference with a given ID and setting for where to get the output from.
   * @param datasetName - The name of the dataset to insert the datapoint into
   * @param inferenceId - The ID of the existing inference to use as a base
   * @param outputKind - How to handle the output field: inherit from inference, use demonstration, or none
   * @returns A promise that resolves with the created datapoint response containing the new ID
   * @throws Error if validation fails or the request fails
   */
  async createDatapointFromInferenceLegacy(
    datasetName: string,
    inferenceId: string,
    outputKind: "inherit" | "demonstration" | "none" = "inherit",
    functionName: string,
    variantName: string,
    episodeId: string,
  ): Promise<DatapointResponse> {
    if (!datasetName || typeof datasetName !== "string") {
      throw new Error("Dataset name must be a non-empty string");
    }

    if (!inferenceId || typeof inferenceId !== "string") {
      throw new Error("Inference ID must be a non-empty string");
    }

    const endpoint = `/internal/datasets/${encodeURIComponent(datasetName)}/datapoints`;

    const request = {
      inference_id: inferenceId,
      output: outputKind,
      function_name: functionName,
      variant_name: variantName,
      episode_id: episodeId,
    };

    const response = await this.fetch(endpoint, {
      method: "POST",
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }

    const body = await response.json();
    return DatapointResponseSchema.parse(body);
  }

  /**
   * Updates an existing datapoint in a dataset.
   * This operation creates a new datapoint with a new ID and marks the old one as stale.
   * The v1 endpoint automatically handles both creating the new version and staling the old one.
   * @param datasetName - The name of the dataset containing the datapoint
   * @param updateDatapointRequest - The update request containing type, id, input, output, and optional fields
   * @returns A promise that resolves with the response containing the new datapoint ID
   * @throws Error if the dataset name is invalid or the request fails
   */
  async updateDatapoint(
    datasetName: string,
    updateDatapointRequest: UpdateDatapointRequest,
  ): Promise<DatapointResponse> {
    const endpoint = `/v1/datasets/${encodeURIComponent(datasetName)}/datapoints`;

    const requestBody: UpdateDatapointsRequest = {
      datapoints: [updateDatapointRequest],
    };

    const response = await this.fetch(endpoint, {
      method: "PATCH",
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }

    const body = (await response.json()) as UpdateDatapointsResponse;
    return { id: body.ids[0] };
  }

  async getDatapoint(
    datapointId: string,
    datasetName?: string,
  ): Promise<Datapoint | undefined> {
    // We currently maintain 2 endpoints for getting a datapoint, with/without the dataset name.
    const endpoint = datasetName
      ? `/v1/datasets/${encodeURIComponent(datasetName)}/get_datapoints`
      : `/v1/datasets/get_datapoints`;
    const requestBody: GetDatapointsRequest = {
      ids: [datapointId],
    };

    const response = await this.fetch(endpoint, {
      method: "POST",
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }

    const body = (await response.json()) as GetDatapointsResponse;
    if (body.datapoints.length === 0) {
      return undefined;
    }
    return body.datapoints[0];
  }

  async listDatapoints(
    dataset_name: string,
    params: ListDatapointsRequest,
  ): Promise<GetDatapointsResponse> {
    const endpoint = `/v1/datasets/${encodeURIComponent(dataset_name)}/list_datapoints`;

    const response = await this.fetch(endpoint, {
      method: "POST",
      body: JSON.stringify(params),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body = (await response.json()) as GetDatapointsResponse;
    return body;
  }

  async listDatasets(params: {
    function_name?: string;
    limit?: number;
    offset?: number;
  }): Promise<ListDatasetsResponse> {
    const searchParams = new URLSearchParams();

    if (params.function_name) {
      searchParams.append("function_name", params.function_name);
    }
    if (params.limit !== undefined) {
      searchParams.append("limit", params.limit.toString());
    }
    if (params.offset !== undefined) {
      searchParams.append("offset", params.offset.toString());
    }

    const queryString = searchParams.toString();
    const endpoint = `/internal/datasets${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return await response.json();
  }

  async updateDatapointsMetadata(
    datasetName: string,
    datapoints: UpdateDatapointsMetadataRequest,
  ): Promise<UpdateDatapointsResponse> {
    const endpoint = `/v1/datasets/${encodeURIComponent(datasetName)}/datapoints/metadata`;
    const response = await this.fetch(endpoint, {
      method: "PATCH",
      body: JSON.stringify(datapoints),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body = (await response.json()) as UpdateDatapointsResponse;
    return body;
  }

  /**
   * Marks datapoints as deleted in a dataset by setting their `staled_at` timestamp.
   * Marked datapoints will no longer appear in default queries, but are preserved in the database for auditing or recovery purposes.
   * @param datasetName - The name of the dataset containing the datapoints
   * @param datapointIds - Array of datapoint UUIDs to mark as deleted
   * @returns A promise that resolves with the response containing the number of marked datapoints
   * @throws Error if the dataset name is invalid, the IDs array is empty, or the request fails
   */
  async deleteDatapoints(
    datasetName: string,
    datapointIds: string[],
  ): Promise<DeleteDatapointsResponse> {
    const endpoint = `/v1/datasets/${encodeURIComponent(datasetName)}/datapoints`;
    const requestBody: DeleteDatapointsRequest = {
      ids: datapointIds,
    };
    const response = await this.fetch(endpoint, {
      method: "DELETE",
      body: JSON.stringify(requestBody),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body = (await response.json()) as DeleteDatapointsResponse;
    return body;
  }

  /**
   * Clones datapoints to a target dataset, preserving all fields except id and dataset_name.
   * @param targetDatasetName - The name of the target dataset to clone datapoints to
   * @param datapointIds - Array of datapoint UUIDs to clone
   * @returns A promise that resolves with the response containing the new datapoint IDs (null if source not found)
   * @throws Error if the dataset name is invalid or the request fails
   */
  async cloneDatapoints(
    targetDatasetName: string,
    datapointIds: string[],
  ): Promise<CloneDatapointsResponse> {
    const endpoint = `/internal/datasets/${encodeURIComponent(targetDatasetName)}/datapoints/clone`;
    const response = await this.fetch(endpoint, {
      method: "POST",
      body: JSON.stringify({ datapoint_ids: datapointIds }),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as CloneDatapointsResponse;
  }

  /**
   * Creates new datapoints in a dataset manually.
   * @param datasetName - The name of the dataset to create datapoints in
   * @param request - The request containing the datapoints to create
   * @returns A promise that resolves with the response containing the new datapoint IDs
   * @throws Error if the dataset name is invalid or the request fails
   */
  async createDatapoints(
    datasetName: string,
    request: CreateDatapointsRequest,
  ): Promise<CreateDatapointsResponse> {
    const endpoint = `/v1/datasets/${encodeURIComponent(datasetName)}/datapoints`;
    const response = await this.fetch(endpoint, {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as CreateDatapointsResponse;
  }

  /**
   * Fetches datapoint count for a dataset, optionally filtered by function name.
   * @param datasetName - The name of the dataset to get count for
   * @param options - Optional parameters for filtering
   * @param options.functionName - Optional function name to filter by
   * @returns A promise that resolves with the datapoint count
   * @throws Error if the request fails
   */
  async getDatapointCount(
    datasetName: string,
    options?: { functionName?: string },
  ): Promise<GetDatapointCountResponse> {
    const searchParams = new URLSearchParams();
    if (options?.functionName) {
      searchParams.append("function_name", options.functionName);
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/datasets/${encodeURIComponent(datasetName)}/datapoints/count${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetDatapointCountResponse;
  }

  async getObject(storagePath: ZodStoragePath): Promise<string> {
    const endpoint = `/internal/object_storage?storage_path=${encodeURIComponent(JSON.stringify(storagePath))}`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return response.text();
  }

  async status(): Promise<StatusResponse> {
    const response = await this.fetch("/status", { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as StatusResponse;
  }

  /**
   * Lists inferences with optional filtering, pagination, and sorting.
   * Uses the public v1 API endpoint.
   * @param request - The list inferences request parameters
   * @returns A promise that resolves with the inferences response
   * @throws Error if the request fails
   */
  async listInferences(
    request: ListInferencesRequest,
  ): Promise<GetInferencesResponse> {
    const response = await this.fetch("/v1/inferences/list_inferences", {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetInferencesResponse;
  }

  /**
   * Retrieves specific inferences by their IDs.
   * Uses the public v1 API endpoint.
   * @param request - The get inferences request containing IDs and optional filters
   * @returns A promise that resolves with the inferences response
   * @throws Error if the request fails
   */
  async getInferences(
    request: GetInferencesRequest,
  ): Promise<GetInferencesResponse> {
    const response = await this.fetch("/v1/inferences/get_inferences", {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetInferencesResponse;
  }

  /**
   * Fetches the gateway configuration for the UI.
   * @returns A promise that resolves with the UiConfig object
   * @throws Error if the request fails
   */
  async getUiConfig(): Promise<UiConfig> {
    const response = await this.fetch("/internal/ui-config", { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as UiConfig;
  }

  /**
   * Fetches inference statistics for a function, optionally filtered by variant or grouped by variant.
   * @param functionName - The name of the function to get stats for
   * @param options - Optional parameters for filtering or grouping
   * @param options.variantName - Optional variant name to filter by
   * @param options.groupBy - Optional grouping (e.g., "variant" to get counts per variant)
   * @returns A promise that resolves with the inference stats
   * @throws Error if the request fails
   */
  async getInferenceStats(
    functionName: string,
    options?: { variantName?: string; groupBy?: "variant" },
  ): Promise<InferenceStatsResponse> {
    const searchParams = new URLSearchParams();
    if (options?.variantName) {
      searchParams.append("variant_name", options.variantName);
    }
    if (options?.groupBy) {
      searchParams.append("group_by", options.groupBy);
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/functions/${encodeURIComponent(functionName)}/inference-stats${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as InferenceStatsResponse;
  }

  /**
   * Fetches the variants used for a function.
   * @param functionName - The name of the function to get variants for
   * @returns A promise that resolves with the variants used for the function
   * @throws Error if the request fails
   */
  async getUsedVariants(functionName: string): Promise<string[]> {
    const response = await this.getInferenceStats(functionName, {
      groupBy: "variant",
    });

    return (response.stats_by_variant ?? []).map(
      (v: InferenceStatsByVariant) => v.variant_name,
    );
  }

  /**
   * Fetches feedback statistics for a function and metric.
   * @param functionName - The name of the function to get stats for
   * @param metricName - The name of the metric to get stats for (or "demonstration")
   * @param threshold - Optional threshold for float metrics (defaults to 0)
   * @returns A promise that resolves with the feedback and curated inference counts
   * @throws Error if the request fails
   */
  async getFeedbackStats(
    functionName: string,
    metricName: string,
    threshold?: number,
  ): Promise<InferenceWithFeedbackStatsResponse> {
    const searchParams = new URLSearchParams();
    if (threshold !== undefined) {
      searchParams.append("threshold", threshold.toString());
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/functions/${encodeURIComponent(functionName)}/inference-stats/${encodeURIComponent(metricName)}${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as InferenceWithFeedbackStatsResponse;
  }

  /**
   * Fetches function throughput data grouped by variant and time period.
   * @param functionName - The name of the function to get throughput data for
   * @param timeWindow - The time granularity for grouping data (minute, hour, day, week, month, cumulative)
   * @param maxPeriods - Maximum number of time periods to return
   * @returns A promise that resolves with the throughput data
   * @throws Error if the request fails
   */
  async getFunctionThroughputByVariant(
    functionName: string,
    timeWindow: TimeWindow,
    maxPeriods: number,
  ): Promise<GetFunctionThroughputByVariantResponse> {
    const searchParams = new URLSearchParams({
      time_window: timeWindow,
      max_periods: maxPeriods.toString(),
    });
    const endpoint = `/internal/functions/${encodeURIComponent(functionName)}/throughput-by-variant?${searchParams.toString()}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetFunctionThroughputByVariantResponse;
  }

  /**
   * Fetches metrics with feedback for a function, optionally filtered by variant.
   * @param functionName - The name of the function to get metrics for
   * @param variantName - Optional variant name to filter by
   * @returns A promise that resolves with metrics and their feedback counts
   * @throws Error if the request fails
   */
  async getFunctionMetricsWithFeedback(
    functionName: string,
    variantName?: string,
  ): Promise<MetricsWithFeedbackResponse> {
    const searchParams = new URLSearchParams();
    if (variantName) {
      searchParams.append("variant_name", variantName);
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/functions/${encodeURIComponent(functionName)}/metrics${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as MetricsWithFeedbackResponse;
  }

  /**
   * Fetches variant performance statistics for a function and metric.
   * @param functionName - The name of the function to get performance stats for
   * @param metricName - The name of the metric to compute performance for
   * @param timeWindow - Time granularity for grouping performance data
   * @param variantName - Optional variant name to filter by
   * @returns A promise that resolves with variant performance statistics
   * @throws Error if the request fails
   */
  async getVariantPerformances(
    functionName: string,
    metricName: string,
    timeWindow: TimeWindow,
    variantName?: string,
  ): Promise<VariantPerformancesResponse> {
    const searchParams = new URLSearchParams();
    searchParams.append("metric_name", metricName);
    searchParams.append("time_window", timeWindow);
    if (variantName) {
      searchParams.append("variant_name", variantName);
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/functions/${encodeURIComponent(functionName)}/variant-performances?${queryString}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as VariantPerformancesResponse;
  }

  /**
   * Fetches model inferences for a given inference ID.
   * @param inferenceId - The UUID of the inference to get model inferences for
   * @returns A promise that resolves with the model inferences response
   * @throws Error if the request fails
   */
  async getModelInferences(
    inferenceId: string,
  ): Promise<GetModelInferencesResponse> {
    const endpoint = `/internal/model_inferences/${encodeURIComponent(inferenceId)}`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetModelInferencesResponse;
  }

  /**
   * Counts the number of distinct models used.
   * @returns A promise that resolves with the count of distinct models
   * @throws Error if the request fails
   */
  async countDistinctModelsUsed(): Promise<CountModelsResponse> {
    const response = await this.fetch("/internal/models/count", {
      method: "GET",
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as CountModelsResponse;
  }

  /**
   * Lists evaluation runs with pagination.
   * @param limit - Maximum number of evaluation runs to return (default: 100)
   * @param offset - Number of evaluation runs to skip for pagination (default: 0)
   * @returns A promise that resolves with the list of evaluation runs
   * @throws Error if the request fails
   */
  async listEvaluationRuns(
    limit: number = 100,
    offset: number = 0,
  ): Promise<ListEvaluationRunsResponse> {
    const searchParams = new URLSearchParams();
    searchParams.append("limit", limit.toString());
    searchParams.append("offset", offset.toString());
    const queryString = searchParams.toString();
    const endpoint = `/internal/evaluations/runs${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as ListEvaluationRunsResponse;
  }

  /**
   * Counts the total number of evaluation runs.
   * @returns A promise that resolves with the evaluation run count
   * @throws Error if the request fails
   */
  async countEvaluationRuns(): Promise<number> {
    const response = await this.fetch("/internal/evaluations/run-stats", {
      method: "GET",
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const count_response =
      (await response.json()) as EvaluationRunStatsResponse;
    return Number(count_response.count);
  }

  /**
   * Creates datapoints from inferences based on either specific inference IDs or an inference query.
   * @param datasetName - The name of the dataset to create datapoints in
   * @param request - The request containing either inference IDs or an inference query with filters
   * @returns A promise that resolves with the response containing the new datapoint IDs
   * @throws Error if the request fails
   */
  async createDatapointsFromInferences(
    datasetName: string,
    request: CreateDatapointsFromInferenceRequest,
  ): Promise<CreateDatapointsResponse> {
    const endpoint = `/v1/datasets/${encodeURIComponent(datasetName)}/from_inferences`;
    const response = await this.fetch(endpoint, {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as CreateDatapointsResponse;
  }

  /**
   * Counts unique datapoints across specified evaluation runs.
   * @param functionName - The name of the function being evaluated
   * @param evaluationRunIds - Array of evaluation run IDs
   * @returns A promise that resolves with the datapoint count
   * @throws Error if the request fails
   */
  async countDatapointsForEvaluation(
    functionName: string,
    evaluationRunIds: string[],
  ): Promise<number> {
    const searchParams = new URLSearchParams();
    searchParams.append("function_name", functionName);
    searchParams.append("evaluation_run_ids", evaluationRunIds.join(","));
    const queryString = searchParams.toString();
    const endpoint = `/internal/evaluations/datapoint-count?${queryString}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }

    const result = (await response.json()) as DatapointStatsResponse;
    return Number(result.count);
  }

  /**
   * Gets workflow evaluation projects with pagination.
   * @param limit - Maximum number of projects to return (default: 100)
   * @param offset - Number of projects to skip (default: 0)
   * @returns A promise that resolves with the workflow evaluation projects response
   * @throws Error if the request fails
   */
  async getWorkflowEvaluationProjects(
    limit: number = 100,
    offset: number = 0,
  ): Promise<GetWorkflowEvaluationProjectsResponse> {
    const searchParams = new URLSearchParams();
    searchParams.append("limit", limit.toString());
    searchParams.append("offset", offset.toString());
    const queryString = searchParams.toString();
    const endpoint = `/internal/workflow-evaluations/projects${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetWorkflowEvaluationProjectsResponse;
  }

  /**
   * Counts workflow evaluation projects.
   * @returns A promise that resolves with the workflow evaluation project count
   * @throws Error if the request fails
   */
  async countWorkflowEvaluationProjects(): Promise<number> {
    const response = await this.fetch(
      "/internal/workflow-evaluations/projects/count",
      { method: "GET" },
    );
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body =
      (await response.json()) as GetWorkflowEvaluationProjectCountResponse;
    return body.count;
  }

  /**
   * Searches workflow evaluation runs by project name and/or search query.
   * @param limit - Maximum number of runs to return (default: 100)
   * @param offset - Number of runs to skip (default: 0)
   * @param projectName - Optional project name to filter by
   * @param searchQuery - Optional search query to filter by (searches run name and ID)
   * @returns A promise that resolves with the search results
   * @throws Error if the request fails
   */
  async searchWorkflowEvaluationRuns(
    limit: number = 100,
    offset: number = 0,
    projectName?: string,
    searchQuery?: string,
  ): Promise<SearchWorkflowEvaluationRunsResponse> {
    const searchParams = new URLSearchParams();
    searchParams.append("limit", limit.toString());
    searchParams.append("offset", offset.toString());
    if (projectName) {
      searchParams.append("project_name", projectName);
    }
    if (searchQuery) {
      searchParams.append("q", searchQuery);
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/workflow-evaluations/runs/search?${queryString}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as SearchWorkflowEvaluationRunsResponse;
  }

  /**
   * Lists workflow evaluation runs with episode counts.
   * @param limit - Maximum number of runs to return (default: 100)
   * @param offset - Number of runs to skip (default: 0)
   * @param runId - Optional run ID to filter by
   * @param projectName - Optional project name to filter by
   * @returns A promise that resolves with the workflow evaluation runs response
   * @throws Error if the request fails
   */
  async listWorkflowEvaluationRuns(
    limit: number = 100,
    offset: number = 0,
    runId?: string,
    projectName?: string,
  ): Promise<ListWorkflowEvaluationRunsResponse> {
    const searchParams = new URLSearchParams();
    searchParams.append("limit", limit.toString());
    searchParams.append("offset", offset.toString());
    if (runId) {
      searchParams.append("run_id", runId);
    }
    if (projectName) {
      searchParams.append("project_name", projectName);
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/workflow-evaluations/list-runs?${queryString}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as ListWorkflowEvaluationRunsResponse;
  }

  /**
   * Counts workflow evaluation runs.
   * @returns A promise that resolves with the workflow evaluation run count
   * @throws Error if the request fails
   */
  async countWorkflowEvaluationRuns(): Promise<number> {
    const response = await this.fetch(
      "/internal/workflow-evaluations/runs/count",
      { method: "GET" },
    );
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body = (await response.json()) as CountWorkflowEvaluationRunsResponse;
    return body.count;
  }

  /**
   * Gets workflow evaluation runs by their IDs.
   * @param runIds - Array of run IDs to fetch
   * @param projectName - Optional project name to filter by
   * @returns A promise that resolves with the workflow evaluation runs response
   * @throws Error if the request fails
   */
  async getWorkflowEvaluationRuns(
    runIds: string[],
    projectName?: string,
  ): Promise<GetWorkflowEvaluationRunsResponse> {
    const searchParams = new URLSearchParams();
    searchParams.append("run_ids", runIds.join(","));
    if (projectName) {
      searchParams.append("project_name", projectName);
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/workflow-evaluations/get-runs?${queryString}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetWorkflowEvaluationRunsResponse;
  }

  /**
   * Fetches statistics for a workflow evaluation run, grouped by metric name.
   * @param runId - The ID of the workflow evaluation run
   * @param metricName - Optional metric name to filter by
   * @returns A promise that resolves with the workflow evaluation run statistics response
   * @throws Error if the request fails
   */
  async getWorkflowEvaluationRunStatistics(
    runId: string,
    metricName?: string,
  ): Promise<GetWorkflowEvaluationRunStatisticsResponse> {
    const searchParams = new URLSearchParams();
    searchParams.append("run_id", runId);
    if (metricName) {
      searchParams.append("metric_name", metricName);
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/workflow-evaluations/run-statistics?${queryString}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetWorkflowEvaluationRunStatisticsResponse;
  }

  /**
   * Lists workflow evaluation run episodes grouped by task name.
   *
   * Returns episodes grouped by task_name. Episodes with NULL task_name are grouped
   * individually using a generated key based on their episode_id.
   *
   * @param runIds - List of run IDs to filter episodes by
   * @param limit - Maximum number of groups to return (default: 15)
   * @param offset - Number of groups to skip (default: 0)
   * @returns A promise that resolves with the grouped episodes response
   * @throws Error if the request fails
   */
  async listWorkflowEvaluationRunEpisodesByTaskName(
    runIds: string[],
    limit: number = 15,
    offset: number = 0,
  ): Promise<ListWorkflowEvaluationRunEpisodesByTaskNameResponse> {
    const searchParams = new URLSearchParams();
    if (runIds.length > 0) {
      searchParams.append("run_ids", runIds.join(","));
    }
    searchParams.append("limit", limit.toString());
    searchParams.append("offset", offset.toString());
    const queryString = searchParams.toString();
    const endpoint = `/internal/workflow-evaluations/episodes-by-task-name${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as ListWorkflowEvaluationRunEpisodesByTaskNameResponse;
  }

  /**
   * Counts the number of distinct episode groups (by task_name) for the given run IDs.
   *
   * Episodes with NULL task_name are counted as individual groups.
   *
   * @param runIds - List of run IDs to filter episodes by
   * @returns A promise that resolves with the count of episode groups
   * @throws Error if the request fails
   */
  async countWorkflowEvaluationRunEpisodeGroupsByTaskName(
    runIds: string[],
  ): Promise<number> {
    const searchParams = new URLSearchParams();
    if (runIds.length > 0) {
      searchParams.append("run_ids", runIds.join(","));
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/workflow-evaluations/episodes-by-task-name/count${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body =
      (await response.json()) as CountWorkflowEvaluationRunEpisodesByTaskNameResponse;
    return body.count;
  }

  /**
   * Gets workflow evaluation run episodes with their feedback for a specific run.
   * @param runId - The run ID to get episodes for
   * @param limit - Maximum number of episodes to return (default: 15)
   * @param offset - Offset for pagination (default: 0)
   * @returns A promise that resolves with the workflow evaluation run episodes response
   * @throws Error if the request fails
   */
  async getWorkflowEvaluationRunEpisodesWithFeedback(
    runId: string,
    limit: number = 15,
    offset: number = 0,
  ): Promise<GetWorkflowEvaluationRunEpisodesWithFeedbackResponse> {
    const searchParams = new URLSearchParams();
    searchParams.append("run_id", runId);
    searchParams.append("limit", limit.toString());
    searchParams.append("offset", offset.toString());
    const queryString = searchParams.toString();
    const endpoint = `/internal/workflow-evaluations/run-episodes?${queryString}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetWorkflowEvaluationRunEpisodesWithFeedbackResponse;
  }

  /**
   * Counts the total number of episodes for a workflow evaluation run.
   * @param runId - The run ID to count episodes for
   * @returns A promise that resolves with the count of episodes
   * @throws Error if the request fails
   */
  async countWorkflowEvaluationRunEpisodes(runId: string): Promise<number> {
    const searchParams = new URLSearchParams();
    searchParams.append("run_id", runId);
    const queryString = searchParams.toString();
    const endpoint = `/internal/workflow-evaluations/run-episodes/count?${queryString}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body =
      (await response.json()) as CountWorkflowEvaluationRunEpisodesResponse;
    return body.count;
  }

  /**
   * Lists inference metadata with optional cursor-based pagination and filtering.
   * @param params - Optional pagination and filter parameters
   * @param params.before - Cursor to fetch records before this ID (mutually exclusive with after)
   * @param params.after - Cursor to fetch records after this ID (mutually exclusive with before)
   * @param params.limit - Maximum number of records to return
   * @param params.function_name - Optional function name to filter by
   * @param params.variant_name - Optional variant name to filter by
   * @param params.episode_id - Optional episode ID to filter by
   * @returns A promise that resolves with the inference metadata response
   * @throws Error if the request fails
   */
  async listInferenceMetadata(params?: {
    before?: string;
    after?: string;
    limit?: number;
    function_name?: string | null;
    variant_name?: string | null;
    episode_id?: string | null;
  }): Promise<ListInferenceMetadataResponse> {
    const searchParams = new URLSearchParams();
    if (params?.before) {
      searchParams.append("before", params.before);
    }
    if (params?.after) {
      searchParams.append("after", params.after);
    }
    if (params?.limit !== undefined) {
      searchParams.append("limit", params.limit.toString());
    }
    if (params?.function_name) {
      searchParams.append("function_name", params.function_name);
    }
    if (params?.variant_name) {
      searchParams.append("variant_name", params.variant_name);
    }
    if (params?.episode_id) {
      searchParams.append("episode_id", params.episode_id);
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/inference_metadata${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as ListInferenceMetadataResponse;
  }

  /**
   * Lists episodes with pagination support.
   * @param limit - Maximum number of episodes to return
   * @param before - Return episodes before this episode_id (for pagination)
   * @param after - Return episodes after this episode_id (for pagination)
   * @returns A promise that resolves with an array of episodes
   * @throws Error if the request fails
   */
  async listEpisodes(
    limit: number,
    before?: string,
    after?: string,
  ): Promise<ListEpisodesResponse> {
    const searchParams = new URLSearchParams();
    searchParams.append("limit", limit.toString());
    if (before) {
      searchParams.append("before", before);
    }
    if (after) {
      searchParams.append("after", after);
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/episodes?${queryString}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as ListEpisodesResponse;
  }

  /**
   * Queries episode table bounds (first_id, last_id, and count).
   * @returns A promise that resolves with the bounds information
   * @throws Error if the request fails
   */
  async queryEpisodeTableBounds(): Promise<TableBoundsWithCount> {
    const endpoint = `/internal/episodes/bounds`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as TableBoundsWithCount;
  }

  /**
   * Counts inferences matching the given parameters.
   * When output_source is "demonstration", only inferences with demonstration feedback are counted.
   * @param request - The count inferences request parameters
   * @returns A promise that resolves with the count of matching inferences
   * @throws Error if the request fails
   */
  async countInferences(request: CountInferencesRequest): Promise<number> {
    const endpoint = "/internal/inferences/count";
    const response = await this.fetch(endpoint, {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const result = (await response.json()) as CountInferencesResponse;
    return Number(result.count);
  }

  /**
   * Gets inference statistics for a specific episode.
   * @param episode_id - The UUID of the episode
   * @returns A promise that resolves with the inference stats
   * @throws Error if the request fails
   */
  async getEpisodeInferenceCount(
    episode_id: string,
  ): Promise<GetEpisodeInferenceCountResponse> {
    const endpoint = `/internal/episodes/${episode_id}/inference-count`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetEpisodeInferenceCountResponse;
  }

  /**
   * Searches evaluation runs by ID or variant name.
   * @param evaluationName - The name of the evaluation
   * @param functionName - The name of the function being evaluated
   * @param query - The search query (case-insensitive)
   * @param limit - Maximum number of results to return (default: 100)
   * @param offset - Number of results to skip (default: 0)
   * @returns A promise that resolves with the search results
   * @throws Error if the request fails
   */
  async searchEvaluationRuns(
    evaluationName: string,
    functionName: string,
    query: string,
    limit: number = 100,
    offset: number = 0,
  ): Promise<SearchEvaluationRunsResponse> {
    const searchParams = new URLSearchParams();
    searchParams.append("evaluation_name", evaluationName);
    searchParams.append("function_name", functionName);
    searchParams.append("query", query);
    searchParams.append("limit", limit.toString());
    searchParams.append("offset", offset.toString());
    const queryString = searchParams.toString();
    const endpoint = `/internal/evaluations/runs/search?${queryString}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }

    return (await response.json()) as SearchEvaluationRunsResponse;
  }

  /**
   * Queries feedback bounds for a given target ID.
   * @param targetId - The target ID (inference_id or episode_id) to query feedback bounds for
   * @returns A promise that resolves with the feedback bounds across all feedback types
   * @throws Error if the request fails
   */
  async getFeedbackBoundsByTargetId(
    targetId: string,
  ): Promise<GetFeedbackBoundsResponse> {
    const endpoint = `/internal/feedback/${encodeURIComponent(targetId)}/bounds`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetFeedbackBoundsResponse;
  }

  /**
   * Queries the latest feedback ID for each metric for a given target.
   * @param targetId - The target ID (inference_id or episode_id) to query feedback for
   * @returns A promise that resolves with a mapping of metric names to their latest feedback IDs
   * @throws Error if the request fails
   */
  async getLatestFeedbackIdByMetric(
    targetId: string,
  ): Promise<Record<string, string>> {
    const endpoint = `/internal/feedback/${encodeURIComponent(targetId)}/latest-id-by-metric`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body = (await response.json()) as LatestFeedbackIdByMetricResponse;
    // Convert optional values to non-optional (ts-rs generates HashMap as optional, but values are always present)
    return Object.fromEntries(
      Object.entries(body.feedback_id_by_metric).filter(
        (entry): entry is [string, string] => entry[1] !== undefined,
      ),
    );
  }

  /**
   * Queries the count of feedback for a given target ID.
   * @param targetId - The target ID (inference_id or episode_id) to count feedback for
   * @returns A promise that resolves with the feedback count
   * @throws Error if the request fails
   */
  async countFeedbackByTargetId(targetId: string): Promise<number> {
    const endpoint = `/internal/feedback/${encodeURIComponent(targetId)}/count`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body = (await response.json()) as CountFeedbackByTargetIdResponse;
    return Number(body.count);
  }

  /**
   * Gets cumulative feedback time series for a function and metric.
   * @param functionName - The name of the function to get feedback for
   * @param metricName - The name of the metric to get feedback for
   * @param timeWindow - The time window granularity for grouping data
   * @param maxPeriods - Maximum number of time periods to return
   * @param variantNames - Optional array of variant names to filter by
   * @returns A promise that resolves with cumulative feedback time series data
   * @throws Error if the request fails
   */
  async getCumulativeFeedbackTimeseries(params: {
    function_name: string;
    metric_name: string;
    time_window: TimeWindow;
    max_periods: number;
    variant_names?: string[];
  }): Promise<CumulativeFeedbackTimeSeriesPoint[]> {
    const searchParams = new URLSearchParams({
      function_name: params.function_name,
      metric_name: params.metric_name,
      time_window: params.time_window,
      max_periods: params.max_periods.toString(),
    });
    if (params.variant_names && params.variant_names.length > 0) {
      searchParams.append("variant_names", params.variant_names.join(","));
    }
    const endpoint = `/internal/feedback/timeseries?${searchParams.toString()}`;
    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body =
      (await response.json()) as GetCumulativeFeedbackTimeseriesResponse;
    return body.timeseries;
  }

  /**
   * Gets information about specific evaluation runs.
   * @param evaluationRunIds - Array of evaluation run UUIDs to query
   * @param functionName - The name of the function being evaluated
   * @returns A promise that resolves with information about the evaluation runs
   * @throws Error if the request fails
   */
  async getEvaluationRunInfos(
    evaluationRunIds: string[],
    functionName: string,
  ): Promise<GetEvaluationRunInfosResponse> {
    const searchParams = new URLSearchParams();
    searchParams.append("evaluation_run_ids", evaluationRunIds.join(","));
    searchParams.append("function_name", functionName);
    const queryString = searchParams.toString();
    const endpoint = `/internal/evaluations/run-infos?${queryString}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetEvaluationRunInfosResponse;
  }

  /**
   * Gets evaluation run infos for a specific datapoint.
   * @param datapointId - The UUID of the datapoint to query
   * @param functionName - The name of the function being evaluated
   * @returns A promise that resolves with information about evaluation runs that include this datapoint
   * @throws Error if the request fails
   */
  async getEvaluationRunInfosForDatapoint(
    datapointId: string,
    functionName: string,
  ): Promise<GetEvaluationRunInfosResponse> {
    const searchParams = new URLSearchParams();
    searchParams.append("function_name", functionName);
    const queryString = searchParams.toString();
    const endpoint = `/internal/evaluations/datapoints/${encodeURIComponent(datapointId)}/run-infos?${queryString}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetEvaluationRunInfosResponse;
  }

  /**
   * Gets evaluation statistics (aggregated metrics) for specified evaluation runs.
   * @param functionName - The name of the function being evaluated
   * @param functionType - The type of function: "chat" or "json"
   * @param metricNames - Array of metric names to query
   * @param evaluationRunIds - Array of evaluation run UUIDs to query
   * @returns A promise that resolves with aggregated statistics for each run/metric
   * @throws Error if the request fails
   */
  async getEvaluationStatistics(
    functionName: string,
    functionType: "chat" | "json",
    metricNames: string[],
    evaluationRunIds: string[],
  ): Promise<GetEvaluationStatisticsResponse> {
    const searchParams = new URLSearchParams();
    searchParams.append("function_name", functionName);
    searchParams.append("function_type", functionType);
    searchParams.append("metric_names", metricNames.join(","));
    searchParams.append("evaluation_run_ids", evaluationRunIds.join(","));
    const queryString = searchParams.toString();
    const endpoint = `/internal/evaluations/statistics?${queryString}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetEvaluationStatisticsResponse;
  }

  /**
   * Gets paginated evaluation results across one or more evaluation runs.
   * @param evaluationName - The name of the evaluation
   * @param evaluationRunIds - Array of evaluation run UUIDs to query
   * @param options - Optional parameters for filtering and pagination
   * @param options.datapointId - Optional datapoint ID to filter results to a specific datapoint
   * @param options.limit - Maximum number of datapoints to return (default: 100)
   * @param options.offset - Number of datapoints to skip (default: 0)
   * @returns A promise that resolves with the evaluation results
   * @throws Error if the request fails
   */
  async getEvaluationResults(
    evaluationName: string,
    evaluationRunIds: string[],
    options: {
      datapointId?: string;
      limit?: number;
      offset?: number;
    } = {},
  ): Promise<GetEvaluationResultsResponse> {
    const { datapointId, limit = 100, offset = 0 } = options;
    const searchParams = new URLSearchParams();
    searchParams.append("evaluation_name", evaluationName);
    searchParams.append("evaluation_run_ids", evaluationRunIds.join(","));
    if (datapointId) {
      searchParams.append("datapoint_id", datapointId);
    }
    if (limit) {
      searchParams.append("limit", limit.toString());
    }
    if (offset) {
      searchParams.append("offset", offset.toString());
    }
    const queryString = searchParams.toString();
    const endpoint = `/internal/evaluations/results?${queryString}`;

    const response = await this.fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    return (await response.json()) as GetEvaluationResultsResponse;
  }

  private async fetch(
    path: string,
    init: {
      method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
      body?: BodyInit;
      headers?: HeadersInit;
    },
  ) {
    const { method } = init;
    const url = `${this.baseUrl}${path}`;

    // For methods which expect payloads, always pass a body value even when it
    // is empty to deal with consistency issues in various runtimes.
    const expectsPayload =
      method === "POST" || method === "PUT" || method === "PATCH";
    const body = init.body || (expectsPayload ? "" : undefined);
    const headers = new Headers(init.headers);
    if (!headers.has("content-type")) {
      headers.set("content-type", "application/json");
    }

    // Add bearer auth for all endpoints except /status
    if (this.apiKey && path !== "/status") {
      headers.set("authorization", `Bearer ${this.apiKey}`);
    }

    try {
      return await fetch(url, { method, headers, body });
    } catch (error) {
      // Convert network errors (ECONNREFUSED, fetch failed, etc.) to GatewayConnectionError
      throw new GatewayConnectionError(error);
    }
  }

  private async getErrorText(response: Response): Promise<string> {
    if (response.bodyUsed) {
      response = response.clone();
    }
    const responseText = await response.text();
    try {
      const parsed = JSON.parse(responseText);
      return typeof parsed?.error === "string" ? parsed.error : responseText;
    } catch {
      // Invalid JSON; return plain text from response
      return responseText;
    }
  }

  private handleHttpError({
    message,
    response,
  }: {
    message: string;
    response: Response;
  }): never {
    throw new TensorZeroServerError(message, {
      // TODO: Ensure that server errors do not leak sensitive information to
      // the client before exposing the statusText
      // statusText: response.statusText,
      status: response.status,
    });
  }
}
