import * as React from "react";
import { useFetcher, type FetcherFormProps } from "react-router";
import type { SubmitTarget, FetcherSubmitOptions } from "react-router";
import type { ZodDisplayInputMessage } from "~/utils/clickhouse/common";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import type {
  CacheParamsOptions,
  FunctionConfig,
  JsonValue,
  PathWithContents,
  UninitializedVariantInfo,
  VariantInfo,
  ChatTemplates,
  StaticToolConfig,
  ToolChoice,
  Tool,
  ResolvedTomlPathData,
} from "~/types/tensorzero";
import type {
  InputMessageContent as TensorZeroContent,
  ImageContent as TensorZeroImage,
  InputMessage as TensorZeroMessage,
  Input as TensorZeroInput,
} from "~/utils/tensorzero";
import type {
  ZodResolvedFileContent,
  ZodDisplayInputMessageContent,
  ZodDisplayInput,
} from "~/utils/clickhouse/common";
import type { InferenceUsage } from "~/utils/clickhouse/helpers";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";
import type { InferenceResponse } from "~/utils/tensorzero";
import { logger } from "~/utils/logger";
import type {
  ClientInferenceParams,
  Input,
  InputMessage,
  InputMessageContent,
  ContentBlockChatOutput,
  JsonInferenceOutput,
  ChatInferenceDatapoint,
  JsonInferenceDatapoint,
} from "~/types/tensorzero";
import type {
  ZodInput,
  ZodInputMessage,
  ZodInputMessageContent,
} from "~/utils/clickhouse/common";
import { v7 } from "uuid";

interface InferenceActionError {
  message: string;
  caught: unknown;
}

type InferenceActionResponse = {
  info?: VariantResponseInfo;
  raw: InferenceResponse;
};

type InferenceActionContext =
  | { state: "init"; data: null; error: null }
  | { state: "idle"; data: InferenceActionResponse | null; error: null }
  | {
      state: "submitting";
      data: InferenceActionResponse | null;
      error: null | InferenceActionError;
    }
  | {
      state: "loading";
      data: InferenceActionResponse | null;
      error: null | InferenceActionError;
    }
  | {
      state: "error";
      data: (Pick<InferenceActionResponse, "raw"> & { info?: never }) | null;
      error: InferenceActionError;
    };

const ENDPOINT = "/api/tensorzero/inference";

type ActionFetcher = InferenceActionContext & {
  Form: React.FC<Omit<FetcherFormProps, "method" | "encType" | "action">>;
  submit(
    target: SubmitTarget,
    opts?: Omit<FetcherSubmitOptions, "method" | "encType" | "action">,
  ): Promise<void>;
};

/**
 * A wrapper around the `useFetcher` hook to handle POST requests to the
 * inference endpoint.
 */
export function useInferenceActionFetcher() {
  const fetcher = useFetcher<InferenceResponse>();
  /**
   * The fetcher's state gives us the current status of the request alongside
   * its data, but it does so in a generic interface that we still need to parse
   * and interpret for rendering related UI. Because this fetcher is only be
   * used for submitting POST requests to the given endpoint, we can safely add
   * types and provide more specific context based on the shape of our data.
   * This also gives our fetcher two additional states:
   *  - `init`: before a request is actually made
   *  - `error`: when the request fails, or when a successful response contains
   *    an error instead of inference data
   *
   * All of this is derived from the fetcher's state and data so that we can
   * avoid managing any state or synchronization via effects internally.
   */
  const context = React.useMemo<InferenceActionContext>(() => {
    const inferenceOutput = fetcher.data;
    if (inferenceOutput) {
      try {
        // Check if the response contains an error
        if ("error" in inferenceOutput && inferenceOutput.error) {
          return {
            state: fetcher.state === "idle" ? "error" : fetcher.state,
            data: { raw: inferenceOutput },
            error: {
              caught: inferenceOutput.error,
              message: `Inference Failed: ${typeof inferenceOutput.error === "string" ? inferenceOutput.error : JSON.stringify(inferenceOutput.error)}`,
            },
          } satisfies InferenceActionContext;
        }

        return {
          state: fetcher.state,
          data: {
            raw: inferenceOutput,
            info:
              "content" in inferenceOutput
                ? {
                    type: "chat" as const,
                    output: inferenceOutput.content,
                    usage: inferenceOutput.usage,
                  }
                : {
                    type: "json" as const,
                    output: inferenceOutput.output,
                    usage: inferenceOutput.usage,
                  },
          },
          error: null,
        } satisfies InferenceActionContext;
      } catch (error) {
        return {
          state: "error",
          data: { raw: inferenceOutput },
          error: { message: "Failed to process response data", caught: error },
        } satisfies InferenceActionContext;
      }
    } else if (fetcher.state === "idle") {
      return {
        state: "init",
        data: null,
        error: null,
      } satisfies InferenceActionContext;
    } else {
      return {
        state: fetcher.state,
        data: null,
        error: null,
      } satisfies InferenceActionContext;
    }
  }, [fetcher.state, fetcher.data]);

  const submit = React.useCallback<ActionFetcher["submit"]>(
    (target, opts) => {
      const submit = fetcher.submit;
      return submit(target, {
        ...opts,
        method: "POST",
        action: ENDPOINT,
      });
    },
    [fetcher.submit],
  );

  const Form = React.useMemo<ActionFetcher["Form"]>(
    () => (props) => {
      const Form = fetcher.Form;
      return <Form {...props} method="POST" action={ENDPOINT} />;
    },
    [fetcher.Form],
  );

  React.useEffect(() => {
    if (context.error?.caught) {
      logger.error("Error processing response:", context.error.caught);
    }
  }, [context.error]);

  return {
    ...context,
    Form,
    submit,
  } satisfies ActionFetcher;
}

// Convert Input type from a datapoint to the Zod Input type used in the UI
export function datapointInputToZodInput(input: Input): ZodInput {
  return {
    system: input.system ?? undefined,
    messages: input.messages.map(inputMessageToZodInputMessage),
  };
}

function inputMessageToZodInputMessage(message: InputMessage): ZodInputMessage {
  return {
    role: message.role,
    content: message.content.map(inputMessageContentToZodInputMessageContent),
  };
}

function inputMessageContentToZodInputMessageContent(
  content: InputMessageContent,
): ZodInputMessageContent {
  switch (content.type) {
    case "raw_text":
    case "template":
    case "text":
    case "thought":
    case "tool_result":
    case "unknown":
      return content;
    case "tool_call":
      if ("raw_arguments" in content) {
        // This is an InferenceResponseToolCall.
        return {
          type: "tool_call",
          name: content.raw_name,
          arguments: content.raw_arguments,
          id: content.id,
        };
      } else {
        return {
          type: "tool_call",
          name: content.name,
          arguments: content.arguments,
          id: content.id,
        };
      }
    case "file": {
      // Handle the StoragePath conversion properly
      if (content.file_type === "url" || content.file_type === "base64") {
        // The file should've been stored, but here they are not. This shouldn't happen.
        throw new Error(
          "URL and base64 files should not be passed to `tensorZeroStoredContentToInputContent`. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.",
        );
      }
      const storageKind = content.storage_path.kind;
      let convertedKind;

      if (storageKind.type === "s3_compatible") {
        // Ensure bucket_name is not null
        convertedKind = {
          ...storageKind,
          bucket_name: storageKind.bucket_name || "",
        };
      } else {
        convertedKind = storageKind;
      }

      return {
        type: "file",
        file: {
          url: content.source_url ?? null,
          mime_type: content.mime_type,
        },
        storage_path: {
          path: content.storage_path.path,
          kind: convertedKind,
        },
      };
    }
  }
}

interface InferenceActionArgs {
  source: "inference";
  resource: ParsedInferenceRow;
  variant: string;
}

interface InferenceDefaultFunctionActionArgs {
  source: "inference";
  resource: ParsedInferenceRow;
  variant?: undefined;
  model_name: string;
}

interface T0DatapointActionArgs {
  source: "t0_datapoint";
  resource: ChatInferenceDatapoint | JsonInferenceDatapoint;
  variant: string;
}

interface ClickHouseDatapointActionArgs {
  source: "clickhouse_datapoint";
  input: ZodDisplayInput;
  functionName: string;
  allowed_tools?: string[];
  additional_tools?: Array<Tool> | null;
  tool_choice?: ToolChoice | null;
  parallel_tool_calls?: boolean | null;
  output_schema?: JsonValue;
  variant?: string;
  cache_options: CacheParamsOptions;
  editedVariantInfo?: VariantInfo;
  functionConfig: FunctionConfig;
  toolsConfig: { [key in string]?: StaticToolConfig };
}

type ActionArgs =
  | InferenceActionArgs
  | InferenceDefaultFunctionActionArgs
  | T0DatapointActionArgs
  | ClickHouseDatapointActionArgs;

function isDefaultFunctionArgs(
  args: ActionArgs,
): args is InferenceDefaultFunctionActionArgs {
  return (
    args.source === "inference" &&
    args.resource.function_name === DEFAULT_FUNCTION
  );
}

export function prepareInferenceActionRequest(
  args: ActionArgs,
): ClientInferenceParams {
  // Create base ClientInferenceParams with default values
  const baseParams: ClientInferenceParams = {
    function_name: null,
    model_name: null,
    episode_id: null,
    input: { system: undefined, messages: [] },
    stream: null,
    params: {
      chat_completion: {
        temperature: null,
        max_tokens: null,
        seed: null,
        top_p: null,
        presence_penalty: null,
        frequency_penalty: null,
        json_mode: null,
        stop_sequences: null,
      },
    },
    variant_name: null,
    provider_tools: [],
    dryrun: null,
    internal: true,
    tags: {
      "tensorzero::ui": "true",
    },
    output_schema: null,
    credentials: new Map(),
    cache_options: {
      max_age_s: null,
      enabled: "on",
    },
    include_original_response: false,
    internal_dynamic_variant_config: null,
  };

  // Prepare request based on source and function type
  if (isDefaultFunctionArgs(args)) {
    const defaultRequest = prepareDefaultFunctionRequest(
      args.resource,
      args.model_name,
    );
    return { ...baseParams, ...defaultRequest };
  } else if (args.source === "clickhouse_datapoint") {
    // Extract tool parameters from the ClickHouse datapoint args
    const dynamicVariantInfo = args.editedVariantInfo
      ? variantInfoToUninitializedVariantInfo(args.editedVariantInfo)
      : null;

    return {
      ...baseParams,
      function_name: args.functionName,
      input: resolvedInputToInput(args.input),
      variant_name: args.variant || null,
      output_schema: args.output_schema || null,
      tool_choice: args.tool_choice || undefined,
      parallel_tool_calls: args.parallel_tool_calls || undefined,
      additional_tools: args.additional_tools || undefined,
      allowed_tools: args.allowed_tools || undefined,
      cache_options: args.cache_options,
      internal_dynamic_variant_config: dynamicVariantInfo,
    };
  } else if (args.source === "t0_datapoint") {
    // Handle datapoints from tensorzero-node (with StoredInput)
    return {
      ...baseParams,
      function_name: args.resource.function_name,
      input: args.resource.input,
      variant_name: args.variant,
    };
  } else {
    // For other sources, the input is already a DisplayInput
    if (
      args.source === "inference" &&
      args.resource.extra_body &&
      args.resource.extra_body.length > 0
    ) {
      throw new Error("Extra body is not supported for inference in UI.");
    }
    const input = resolvedInputToInput(args.resource.input);
    // TODO: this is unsupported in Node bindings for now
    // const extra_body =
    //   args.source === "inference" ? args.resource.extra_body : undefined;

    return {
      ...baseParams,
      function_name: args.resource.function_name,
      input,
      variant_name: args.variant,
    };
  }
}

function prepareDefaultFunctionRequest(
  inference: ParsedInferenceRow,
  selectedVariant: string,
): Partial<ClientInferenceParams> {
  const input = resolvedInputToInput(inference.input);
  if (inference.function_type === "chat") {
    const tool_choice = inference.tool_params?.tool_choice;
    const parallel_tool_calls = inference.tool_params?.parallel_tool_calls;
    const tools_available = inference.tool_params?.tools_available;
    return {
      model_name: selectedVariant,
      input,
      tool_choice: tool_choice,
      parallel_tool_calls: parallel_tool_calls || undefined,
      // We need to add all tools as additional for the default function
      additional_tools: tools_available,
    };
  } else if (inference.function_type === "json") {
    // This should never happen, just in case and for type safety
    const output_schema = inference.output_schema;
    return {
      model_name: selectedVariant,
      input,
      output_schema: output_schema || null,
    };
  }

  // Fallback case
  return {
    model_name: selectedVariant,
    input,
  };
}

export type VariantResponseInfo =
  | {
      type: "chat";
      output?: ContentBlockChatOutput[];
      usage?: InferenceUsage;
    }
  | {
      type: "json";
      output?: JsonInferenceOutput;
      usage?: InferenceUsage;
    };

export function resolvedInputToInput(input: ZodDisplayInput): Input {
  return {
    system: input.system || null,
    messages: input.messages.map(resolvedInputMessageToInputMessage),
  };
}

export function resolvedInputToTensorZeroInput(
  input: ZodDisplayInput,
): TensorZeroInput {
  return {
    ...input,
    messages: input.messages.map(resolvedInputMessageToTensorZeroMessage),
  };
}

function resolvedInputMessageToTensorZeroMessage(
  message: ZodDisplayInputMessage,
): TensorZeroMessage {
  return {
    ...message,
    content: message.content.map(
      resolvedInputMessageContentToTensorZeroContent,
    ),
  };
}

function resolvedInputMessageContentToTensorZeroContent(
  content: ZodDisplayInputMessageContent,
): TensorZeroContent {
  switch (content.type) {
    case "text":
      return {
        type: "text",
        text: content.text,
      };
    case "missing_function_text":
      return {
        type: "text",
        text: content.value,
      };
    case "raw_text":
    case "tool_call":
    case "tool_result":
    case "thought":
    case "template":
    case "unknown":
      return content;
    case "file":
      return resolvedFileContentToTensorZeroFile(content);
    case "file_error":
      throw new Error("Can't convert image error to tensorzero content");
  }
}

function resolvedFileContentToTensorZeroFile(
  content: ZodResolvedFileContent,
): TensorZeroImage {
  const data = content.file.data.split(",")[1];
  return {
    type: "image",
    mime_type: content.file.mime_type,
    data,
  };
}

function resolvedInputMessageToInputMessage(
  message: ZodDisplayInputMessage,
): InputMessage {
  return {
    role: message.role,
    content: message.content.map(
      resolvedInputMessageContentToInputMessageContent,
    ),
  };
}

function resolvedInputMessageContentToInputMessageContent(
  content: ZodDisplayInputMessageContent,
): InputMessageContent {
  switch (content.type) {
    case "template":
      return content;
    case "text":
      return {
        type: "text",
        text: content.text,
      };
    case "missing_function_text":
      return {
        type: "text",
        text: content.value,
      };
    case "raw_text":
      return {
        type: "raw_text",
        value: content.value,
      };
    case "tool_call": {
      let parsedArguments;
      try {
        parsedArguments = JSON.parse(content.arguments);
      } catch {
        parsedArguments = content.arguments;
      }
      return {
        type: "tool_call",
        id: content.id,
        name: content.name,
        arguments: parsedArguments,
        raw_arguments: content.arguments,
        raw_name: content.name,
      };
    }
    case "tool_result":
      return {
        type: "tool_result",
        id: content.id,
        name: content.name,
        result: content.result,
      };
    case "thought":
      return {
        type: "thought",
        text: content.text,
        signature: content.signature,
        provider_type: content._internal_provider_type,
      };
    case "unknown":
      return {
        type: "unknown",
        data: content.data,
        model_name: content.model_name,
        provider_name: content.provider_name,
      };
    case "file":
      return resolvedFileContentToClientFile(content);
    case "file_error":
      throw new Error("Can't convert image error to client content");
  }
}

function resolvedFileContentToClientFile(
  content: ZodResolvedFileContent,
): InputMessageContent {
  const data = content.file.data.split(",")[1];
  return {
    type: "file",
    file_type: "base64",
    mime_type: content.file.mime_type,
    data,
  };
}

function convertTemplate(
  template: PathWithContents | null,
): ResolvedTomlPathData | null {
  if (!template) return null;
  return {
    __tensorzero_remapped_path: `template_${v7()}`,
    __data: template.contents,
  };
}

function stringToTemplate(
  template: string | null,
): ResolvedTomlPathData | null {
  if (!template) return null;
  return {
    __tensorzero_remapped_path: `template_${v7()}`,
    __data: template,
  };
}

function convertTemplatesToRecord(
  templates: ChatTemplates,
): Record<string, { path: ResolvedTomlPathData }> {
  const result: Record<string, { path: ResolvedTomlPathData }> = {};
  for (const [name, templateData] of Object.entries(templates)) {
    const converted = convertTemplate(templateData?.template || null);
    if (converted) {
      result[name] = { path: converted };
    }
  }
  return result;
}

function variantInfoToUninitializedVariantInfo(
  variantInfo: VariantInfo,
): UninitializedVariantInfo {
  const baseUninitialized = {
    timeouts: variantInfo.timeouts,
  };

  const inner = variantInfo.inner;

  switch (inner.type) {
    case "chat_completion": {
      // Convert all templates
      const templates = convertTemplatesToRecord(inner.templates);

      return {
        ...baseUninitialized,
        type: "chat_completion" as const,
        weight: inner.weight,
        model: inner.model,
        input_wrappers: null,
        // Set legacy fields to null when using new templates format
        system_template: null,
        user_template: null,
        assistant_template: null,
        // New templates field with all templates
        templates,
        temperature: inner.temperature,
        max_tokens: inner.max_tokens,
        seed: inner.seed,
        top_p: inner.top_p,
        presence_penalty: inner.presence_penalty,
        frequency_penalty: inner.frequency_penalty,
        stop_sequences: inner.stop_sequences,
        json_mode: inner.json_mode,
        retries: inner.retries,
      };
    }

    case "best_of_n_sampling": {
      // Convert all evaluator templates
      const evaluatorTemplates = convertTemplatesToRecord(
        inner.evaluator.templates,
      );

      return {
        ...baseUninitialized,
        type: "experimental_best_of_n_sampling" as const,
        weight: inner.weight,
        timeout_s: inner.timeout_s,
        candidates: inner.candidates,
        evaluator: {
          weight: inner.evaluator.weight,
          model: inner.evaluator.model,
          input_wrappers: null,
          // Set legacy fields to null when using new templates format
          system_template: null,
          user_template: null,
          assistant_template: null,
          // New templates field with all templates
          templates: evaluatorTemplates,
          temperature: inner.evaluator.temperature,
          top_p: inner.evaluator.top_p,
          max_tokens: inner.evaluator.max_tokens,
          presence_penalty: inner.evaluator.presence_penalty,
          frequency_penalty: inner.evaluator.frequency_penalty,
          seed: inner.evaluator.seed,
          stop_sequences: inner.evaluator.stop_sequences,
          json_mode: inner.evaluator.json_mode,
          retries: inner.evaluator.retries,
        },
      };
    }

    case "dicl":
      return {
        ...baseUninitialized,
        type: "experimental_dynamic_in_context_learning" as const,
        weight: inner.weight,
        embedding_model: inner.embedding_model,
        k: inner.k,
        model: inner.model,
        system_instructions: stringToTemplate(inner.system_instructions.__data),
        temperature: inner.temperature,
        top_p: inner.top_p,
        stop_sequences: inner.stop_sequences,
        presence_penalty: inner.presence_penalty,
        frequency_penalty: inner.frequency_penalty,
        max_tokens: inner.max_tokens,
        seed: inner.seed,
        json_mode: inner.json_mode,
        retries: inner.retries,
        max_distance: inner.max_distance,
      };

    case "mixture_of_n": {
      // Convert all fuser templates
      const fuserTemplates = convertTemplatesToRecord(inner.fuser.templates);

      return {
        ...baseUninitialized,
        type: "experimental_mixture_of_n" as const,
        weight: inner.weight,
        timeout_s: inner.timeout_s,
        candidates: inner.candidates,
        fuser: {
          weight: inner.fuser.weight,
          model: inner.fuser.model,
          input_wrappers: null,
          // Set legacy fields to null when using new templates format
          system_template: null,
          user_template: null,
          assistant_template: null,
          // New templates field with all templates
          templates: fuserTemplates,
          temperature: inner.fuser.temperature,
          top_p: inner.fuser.top_p,
          max_tokens: inner.fuser.max_tokens,
          presence_penalty: inner.fuser.presence_penalty,
          frequency_penalty: inner.fuser.frequency_penalty,
          seed: inner.fuser.seed,
          stop_sequences: inner.fuser.stop_sequences,
          json_mode: inner.fuser.json_mode,
          retries: inner.fuser.retries,
        },
      };
    }

    case "chain_of_thought": {
      // Convert all templates
      const templates = convertTemplatesToRecord(inner.templates);

      return {
        ...baseUninitialized,
        type: "experimental_chain_of_thought" as const,
        weight: inner.weight,
        model: inner.model,
        input_wrappers: null,
        // Set legacy fields to null when using new templates format
        system_template: null,
        user_template: null,
        assistant_template: null,
        // New templates field with all templates
        templates,
        temperature: inner.temperature,
        top_p: inner.top_p,
        max_tokens: inner.max_tokens,
        presence_penalty: inner.presence_penalty,
        frequency_penalty: inner.frequency_penalty,
        seed: inner.seed,
        stop_sequences: inner.stop_sequences,
        json_mode: inner.json_mode,
        retries: inner.retries,
      };
    }

    default:
      throw new Error(`Unknown variant type`);
  }
}
