/*
TensorZero Client (for internal use only for now)

TODO(shuyangli): Figure out a way to generate the HTTP client, possibly from Schema.
*/

import { z } from "zod";
import {
  contentBlockChatOutputSchema,
  thoughtContentSchema,
  JsonValueSchema,
  type StoragePath,
} from "~/utils/clickhouse/common";
import { TensorZeroServerError } from "./errors";
import type {
  Datapoint as TensorZeroDatapoint,
  UpdateDatapointsMetadataRequest,
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
  arguments: JsonValueSchema,
});

export const TemplateContentSchema = z.object({
  type: z.literal("template"),
  name: z.string(),
  arguments: JsonValueSchema,
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
  data: JsonValueSchema,
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
  system: JsonValueSchema.optional(),
  messages: z.array(InputMessageSchema),
});
export type Input = z.infer<typeof InputSchema>;

/**
 * A Tool that the LLM may call.
 */
export const ToolSchema = z.object({
  description: z.string(),
  parameters: JsonValueSchema,
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
export const InferenceParamsSchema = z.record(z.record(JsonValueSchema));
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
  output_schema: JsonValueSchema.optional(),
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
    parsed: JsonValueSchema.nullable(),
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
  value: JsonValueSchema,
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
export const ToolParamsSchema = z.record(JsonValueSchema);
export type ToolParams = z.infer<typeof ToolParamsSchema>;

/**
 * Base schema for datapoints with common fields
 */
const BaseDatapointSchema = z.object({
  function_name: z.string(),
  id: z.string().uuid(),
  episode_id: z.string().uuid().nullish(),
  input: InputSchema,
  output: JsonValueSchema,
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
  output_schema: JsonValueSchema,
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
export type Datapoint = z.infer<typeof DatapointSchema>;

/**
 * Schema for datapoint response
 */
export const DatapointResponseSchema = z.object({
  id: z.string(),
});
export type DatapointResponse = z.infer<typeof DatapointResponseSchema>;

/**
 * Schema for status response
 */
export const StatusResponseSchema = z.object({
  status: z.string(),
  version: z.string(),
});
export type StatusResponse = z.infer<typeof StatusResponseSchema>;

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

  // Overloads for inference:
  /*
  This is deprecated in favor of the native client
  async inference(
    request: InferenceRequest & { stream?: false | undefined },
  ): Promise<InferenceResponse>;
  inference(
    request: InferenceRequest & { stream: true },
  ): Promise<AsyncGenerator<InferenceResponse, void, unknown>>;
  async inference(
    request: InferenceRequest,
  ): Promise<
    InferenceResponse | AsyncGenerator<InferenceResponse, void, unknown>
  > {
    if (request.stream) {
      // Return an async generator that yields each SSE event as an InferenceResponse.
      return this.inferenceStream(request);
    } else {
      const response = await this.fetch("/inference", {
        method: "POST",
        body: JSON.stringify(request),
      });
      if (!response.ok) {
        const message = await this.getErrorText(response);
        this.handleHttpError({ message, response });
      }
      return (await response.json()) as InferenceResponse;
    }
  }
   * Returns an async generator that yields inference responses as they arrive via SSE.
   *
   * Note: The TensorZero gateway streams responses as Server-Sent Events (SSE). This simple parser
   * splits events by a double newline. Adjust if the event format changes.
  private async *inferenceStream(
    request: InferenceRequest,
  ): AsyncGenerator<InferenceResponse, void, unknown> {
    const response = await this.fetch("/inference", {
      method: "POST",
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    if (!response.body) {
      this.handleHttpError({
        message: `Streaming inference failed; response body is not readable`,
        response,
      });
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const part of parts) {
        const lines = part.split("\n").map((line) => line.trim());
        let dataStr = "";
        for (const line of lines) {
          if (line.startsWith("data:")) {
          // note the line below has  an escape backslash for the comment to work
          dataStr += line.replace(/^data:\s*\/, "");
          }
        }
        if (dataStr === "[DONE]") {
          return;
        }
        if (dataStr) {
          try {
            const parsed = JSON.parse(dataStr);
            yield parsed as InferenceResponse;
          } catch (err) {
            logger.error("Failed to parse SSE data:", err);
          }
        }
      }
    }
  }
  */

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
  async createDatapoint(
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
   * Updates an existing datapoint in a dataset with the given ID.
   * @param datasetName - The name of the dataset containing the datapoint
   * @param datapointId - The UUID of the datapoint to update
   * @param datapoint - The datapoint data containing function_name, input, output, and optional fields
   * @returns A promise that resolves with the response containing the datapoint ID
   * @throws Error if validation fails or the request fails
   */
  async updateDatapoint(
    datasetName: string,
    datapoint: Datapoint,
  ): Promise<DatapointResponse> {
    // TODO(#3921): Move to native Rust client.
    if (!datasetName || typeof datasetName !== "string") {
      throw new Error("Dataset name must be a non-empty string");
    }

    // Validate the datapoint using the Zod schema
    const validationResult = DatapointSchema.safeParse(datapoint);
    if (!validationResult.success) {
      throw new Error(`Invalid datapoint: ${validationResult.error.message}`);
    }

    const endpoint = `/internal/datasets/${encodeURIComponent(datasetName)}/datapoints/${encodeURIComponent(datapoint.id)}`;
    // We need to remove the id field from the datapoint before sending it to the server
    const { id, ...rest } = datapoint;

    const response = await this.fetch(endpoint, {
      method: "PUT",
      body: JSON.stringify(rest),
    });

    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }

    const body = await response.json();
    return DatapointResponseSchema.parse(body);
  }

  async listDatapoints(
    dataset_name: string,
    function_name?: string,
    limit?: number,
    offset?: number,
  ): Promise<TensorZeroDatapoint[]> {
    const params = new URLSearchParams();
    if (function_name) {
      params.append("function_name", function_name);
    }
    if (limit !== undefined) {
      params.append("limit", limit.toString());
    }
    if (offset !== undefined) {
      params.append("offset", offset.toString());
    }

    const queryString = params.toString();
    const endpoint = `/datasets/${encodeURIComponent(dataset_name)}/datapoints${queryString ? `?${queryString}` : ""}`;

    const response = await this.fetch(endpoint, {
      method: "GET",
    });
    if (!response.ok) {
      const message = await this.getErrorText(response);
      this.handleHttpError({ message, response });
    }
    const body = await response.json();
    return body as TensorZeroDatapoint[];
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

  async getObject(storagePath: StoragePath): Promise<string> {
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
    return StatusResponseSchema.parse(await response.json());
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

    return await fetch(url, { method, headers, body });
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
