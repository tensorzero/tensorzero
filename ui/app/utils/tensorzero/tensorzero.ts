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
  CreateDatapointsRequest,
  CreateDatapointsResponse,
  Datapoint,
  DeleteDatapointsRequest,
  DeleteDatapointsResponse,
  GetDatapointsRequest,
  GetDatapointsResponse,
  GetInferencesResponse,
  InferenceStatsResponse,
  ListDatapointsRequest,
  ListDatasetsResponse,
  ListInferencesRequest,
  StatusResponse,
  UiConfig,
  UpdateDatapointRequest,
  UpdateDatapointsMetadataRequest,
  UpdateDatapointsRequest,
  UpdateDatapointsResponse,
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
 * Schema for tool parameters in a datapoint
 */
export const ToolParamsSchema = z.record(ZodJsonValueSchema);
export type ToolParams = z.infer<typeof ToolParamsSchema>;

/**
 * Base schema for datapoints with common fields
 */
const BaseDatapointSchema = z.object({
  function_name: z.string(),
  id: z.string().uuid(),
  episode_id: z.string().uuid().nullish(),
  input: InputSchema,
  output: ZodJsonValueSchema,
  tags: z.record(z.string()).optional(),
  auxiliary: z.string().optional(),
  is_custom: z.boolean(),
  source_inference_id: z.string().uuid().nullish(),
  name: z.string().nullish(),
  staled_at: z.string().datetime().nullish(),
});

/**
 * Schema for chat inference datapoints
 */
export const ChatInferenceDatapointSchema = BaseDatapointSchema.extend({
  tool_params: ToolParamsSchema.optional(),
});
export type ChatInferenceDatapoint = z.infer<
  typeof ChatInferenceDatapointSchema
>;

/**
 * Schema for JSON inference datapoints
 */
export const JsonInferenceDatapointSchema = BaseDatapointSchema.extend({
  output_schema: ZodJsonValueSchema,
});
export type JsonInferenceDatapoint = z.infer<
  typeof JsonInferenceDatapointSchema
>;

/**
 * Combined schema for any type of datapoint
 */
export const DatapointSchema = z.union([
  ChatInferenceDatapointSchema,
  JsonInferenceDatapointSchema,
]);
export type ZodDatapoint = z.infer<typeof DatapointSchema>;

/**
 * Schema for datapoint response
 */
export const DatapointResponseSchema = z.object({
  id: z.string(),
});
export type DatapointResponse = z.infer<typeof DatapointResponseSchema>;

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
   * Fetches inference statistics for a function, optionally filtered by variant.
   * @param functionName - The name of the function to get stats for
   * @param variantName - Optional variant name to filter by
   * @returns A promise that resolves with the inference count
   * @throws Error if the request fails
   */
  async getInferenceStats(
    functionName: string,
    variantName?: string,
  ): Promise<InferenceStatsResponse> {
    const searchParams = new URLSearchParams();
    if (variantName) {
      searchParams.append("variant_name", variantName);
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
