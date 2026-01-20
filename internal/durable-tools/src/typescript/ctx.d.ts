/**
 * Type definitions for the global `ctx` object available in TypeScript tools.
 *
 * The `ctx` object provides access to ToolContext methods for:
 * - Task/episode identity
 * - Calling and spawning other tools
 * - Making inference requests
 * - Durable primitives (rand, now, uuid7, sleep)
 * - Event handling
 */

/**
 * Parameters for making an inference request.
 */
interface InferenceParams {
  /** The function name. Exactly one of `function_name` or `model_name` must be provided. */
  function_name?: string;
  /** The model name to run using a default function. Exactly one of `function_name` or `model_name` must be provided. */
  model_name?: string;
  /** The episode ID (if not provided, it'll be set to inference_id). */
  episode_id?: string;
  /** The input for the inference. */
  input: InferenceInput;
  /** Whether to stream the response. Default false. */
  stream?: boolean;
  /** Inference-time parameter overrides. */
  params?: Record<string, unknown>;
  /** Pin a specific variant to be used. */
  variant_name?: string;
  /** If true, the inference will not be stored. */
  dryrun?: boolean;
  /** If true, the inference will be internal. */
  internal?: boolean;
  /** Tags to add to the inference. */
  tags?: Record<string, string>;
  /** Tool-related parameters. */
  allowed_tools?: string[];
  additional_tools?: unknown[];
  tool_choice?: unknown;
  parallel_tool_calls?: boolean;
  /** Output schema for JSON inference. */
  output_schema?: unknown;
  /** Credentials for the inference. */
  credentials?: Record<string, string>;
  /** Cache options. */
  cache_options?: Record<string, unknown>;
  /** Include the original response from the model. */
  include_original_response?: boolean;
  /** Include raw usage data. */
  include_raw_usage?: boolean;
}

/**
 * Input for an inference request.
 */
interface InferenceInput {
  /** System message or messages. */
  system?: string | SystemMessage | SystemMessage[];
  /** User/assistant message history. */
  messages?: Message[];
}

/**
 * A system message.
 */
interface SystemMessage {
  /** The content of the system message. */
  content: string;
}

/**
 * A message in the conversation.
 */
interface Message {
  /** The role of the message sender. */
  role: "user" | "assistant";
  /** The content of the message. Can be a string or structured content. */
  content: string | MessageContent[];
}

/**
 * Structured message content.
 */
type MessageContent =
  | { type: "text"; text: string }
  | { type: "tool_call"; id: string; name: string; arguments: string }
  | { type: "tool_result"; id: string; name: string; result: string };

/**
 * Result from an inference request.
 */
interface InferenceResult {
  /** The unique inference ID. */
  inference_id: string;
  /** The episode ID. */
  episode_id: string;
  /** The variant that was used. */
  variant_name: string;
  /** The output from the inference. */
  output: InferenceOutput;
  /** Usage information. */
  usage?: UsageInfo;
  /** The original response from the model, if requested. */
  original_response?: string;
}

/**
 * Output from an inference.
 */
type InferenceOutput =
  | { type: "chat"; content: ChatContent[] }
  | { type: "json"; raw: string; parsed?: unknown };

/**
 * Chat content in an inference output.
 */
type ChatContent =
  | { type: "text"; text: string }
  | { type: "tool_call"; id: string; name: string; raw_name: string; raw_arguments: string; arguments?: unknown };

/**
 * Usage information from an inference.
 */
interface UsageInfo {
  input_tokens?: number;
  output_tokens?: number;
}

/**
 * The global context object providing access to durable tool operations.
 */
interface Ctx {
  /**
   * Get the current task ID.
   * @returns The task ID as a string.
   */
  taskId(): string;

  /**
   * Get the current episode ID.
   * @returns The episode ID as a string.
   */
  episodeId(): string;

  /**
   * Call another tool and wait for its result.
   * @param name - The name of the tool to call.
   * @param llmParams - Parameters from the LLM for the tool.
   * @param sideInfo - Additional side information for the tool.
   * @returns A promise resolving to the tool's output.
   */
  callTool<T = unknown>(
    name: string,
    llmParams: unknown,
    sideInfo: unknown
  ): Promise<T>;

  /**
   * Spawn a tool to run in the background.
   * @param name - The name of the tool to spawn.
   * @param llmParams - Parameters from the LLM for the tool.
   * @param sideInfo - Additional side information for the tool.
   * @returns A promise resolving to a handle ID for joining later.
   */
  spawnTool(
    name: string,
    llmParams: unknown,
    sideInfo: unknown
  ): Promise<string>;

  /**
   * Wait for a previously spawned tool to complete and get its result.
   * @param handleId - The handle ID returned by spawnTool.
   * @returns A promise resolving to the tool's output.
   */
  joinTool<T = unknown>(handleId: string): Promise<T>;

  /**
   * Make an inference request.
   * @param params - The inference parameters.
   * @returns A promise resolving to the inference result.
   */
  inference(params: InferenceParams): Promise<InferenceResult>;

  /**
   * Get a durable random number between 0 and 1.
   * This value is deterministic on replay.
   * @returns A promise resolving to a random number.
   */
  rand(): Promise<number>;

  /**
   * Get the current durable timestamp.
   * This value is deterministic on replay.
   * @returns A promise resolving to an RFC 3339 formatted timestamp string.
   */
  now(): Promise<string>;

  /**
   * Generate a durable UUID v7.
   * This value is deterministic on replay.
   * @returns A promise resolving to a UUID string.
   */
  uuid7(): Promise<string>;

  /**
   * Sleep for a durable duration.
   * The sleep is deterministic on replay.
   * @param name - A unique name for this sleep operation.
   * @param durationMs - The duration to sleep in milliseconds.
   * @returns A promise that resolves after the duration.
   */
  sleepFor(name: string, durationMs: number): Promise<void>;

  /**
   * Wait for an external event.
   * @param eventName - The name of the event to wait for.
   * @param timeoutMs - Optional timeout in milliseconds.
   * @returns A promise resolving to the event payload.
   */
  awaitEvent<T = unknown>(
    eventName: string,
    timeoutMs?: number
  ): Promise<T>;

  /**
   * Emit an event.
   * @param eventName - The name of the event to emit.
   * @param payload - The event payload.
   * @returns A promise that resolves when the event is emitted.
   */
  emitEvent(eventName: string, payload: unknown): Promise<void>;
}

/**
 * The global context object.
 */
declare const ctx: Ctx;
