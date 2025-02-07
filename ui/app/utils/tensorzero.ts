/*
TensorZero Client (for internal use only for now)
*/

/**
 * JSON types.
 */
export type JSONValue =
  | string
  | number
  | boolean
  | null
  | JSONObject
  | JSONArray;
export interface JSONObject {
  [key: string]: JSONValue;
}
export type JSONArray = Array<JSONValue>;

/**
 * Roles for input messages.
 */
export type Role = "system" | "user" | "assistant" | "tool";

/**
 * An input message’s content may be structured.
 */
export type InputMessageContent =
  | { type: "text"; value: JSONValue }
  | { type: "raw_text"; value: string }
  | { type: "tool_call"; value: ToolCall }
  | { type: "tool_result"; value: ToolResult };

/**
 * An input message sent by the client.
 */
export interface InputMessage {
  role: Role;
  content: InputMessageContent[];
}

/**
 * The inference input object.
 */
export interface Input {
  system?: JSONValue;
  messages: InputMessage[];
}

/**
 * A Tool that the LLM may call.
 */
export interface Tool {
  description: string;
  parameters: JSONValue;
  name: string;
  strict?: boolean;
}

/**
 * A tool call request.
 */
export interface ToolCall {
  name: string;
  /** The arguments as a JSON string. */
  arguments: string;
  id: string;
}

/**
 * A tool call result.
 */
export interface ToolResult {
  name: string;
  result: string;
  id: string;
}

/**
 * Tool choice, which controls how tools are selected.
 * This mirrors the Rust enum:
 * - "none": no tool should be used
 * - "auto": let the model decide
 * - "required": the model must call a tool
 * - { specific: "tool_name" }: force a specific tool
 */
export type ToolChoice = "none" | "auto" | "required" | { specific: string };

/**
 * Inference parameters allow runtime overrides for a given variant.
 */
export type InferenceParams = Record<string, JSONObject>;

/**
 * The request type for inference. These fields correspond roughly
 * to the Rust `Params` struct.
 *
 * Exactly one of `function_name` or `model_name` should be provided.
 */
export interface InferenceRequest {
  function_name?: string;
  model_name?: string;
  episode_id?: string;
  input: Input;
  stream?: boolean;
  params?: InferenceParams;
  variant_name?: string;
  dryrun?: boolean;
  tags?: Record<string, string>;
  allowed_tools?: string[];
  additional_tools?: Tool[];
  tool_choice?: ToolChoice;
  parallel_tool_calls?: boolean;
  output_schema?: JSONValue;
  credentials?: Record<string, string>;
}

/**
 * For chat functions the API returns a list of content blocks.
 */
export type ContentBlock =
  | { type: "text"; text: string }
  | {
      type: "tool_call";
      id: string;
      name: string | null;
      arguments: JSONObject | null;
      raw_arguments: string;
      raw_name: string;
    };

/**
 * Inference responses vary based on the function type.
 */
export interface ChatInferenceResponse {
  inference_id: string;
  episode_id: string;
  variant_name: string;
  content: ContentBlock[];
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

export interface JSONInferenceResponse {
  inference_id: string;
  episode_id: string;
  variant_name: string;
  output: {
    raw: string;
    parsed: JSONValue | null;
  };
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

/**
 * The overall inference response is a union of chat and JSON responses.
 */
export type InferenceResponse = ChatInferenceResponse | JSONInferenceResponse;

/**
 * Feedback requests attach a metric value to a given inference or episode.
 */
export interface FeedbackRequest {
  dryrun?: boolean;
  episode_id?: string;
  inference_id?: string;
  metric_name: string;
  tags?: Record<string, string>;
  value: JSONValue;
}

export interface FeedbackResponse {
  feedback_id: string;
}

/**
 * A client for calling the TensorZero Gateway inference and feedback endpoints.
 */
export class TensorZeroClient {
  private baseUrl: string;

  /**
   * @param baseUrl - The base URL of the TensorZero Gateway (e.g. "http://localhost:3000")
   */
  constructor(baseUrl: string) {
    // Remove any trailing slash for consistency.
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  // Overloads for inference:
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
    const url = `${this.baseUrl}/inference`;
    if (request.stream) {
      // Return an async generator that yields each SSE event as an InferenceResponse.
      return this.inferenceStream(request);
    } else {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });
      if (!response.ok) {
        throw new Error(
          `Inference request failed with status ${response.status}`,
        );
      }
      return (await response.json()) as InferenceResponse;
    }
  }

  /**
   * Returns an async generator that yields inference responses as they arrive via SSE.
   *
   * Note: The TensorZero gateway streams responses as Server-Sent Events (SSE). This simple parser
   * splits events by a double newline. Adjust if the event format changes.
   */
  private async *inferenceStream(
    request: InferenceRequest,
  ): AsyncGenerator<InferenceResponse, void, unknown> {
    const url = `${this.baseUrl}/inference`;
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok || !response.body) {
      throw new Error(
        `Streaming inference failed with status ${response.status}`,
      );
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
            dataStr += line.replace(/^data:\s*/, "");
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
            console.error("Failed to parse SSE data:", err);
          }
        }
      }
    }
  }

  /**
   * Sends feedback for a particular inference or episode.
   * @param request - The feedback request payload.
   * @returns A promise that resolves with the feedback response.
   */
  async feedback(request: FeedbackRequest): Promise<FeedbackResponse> {
    const url = `${this.baseUrl}/feedback`;
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      throw new Error(`Feedback request failed with status ${response.status}`);
    }
    return (await response.json()) as FeedbackResponse;
  }
}
