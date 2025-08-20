import * as React from "react";
import { useFetcher, type FetcherFormProps } from "react-router";
import type { SubmitTarget, FetcherSubmitOptions } from "react-router";
import type { DisplayInputMessage } from "~/utils/clickhouse/common";
import type {
  CacheParamsOptions,
  ClientInput,
  ClientInputMessage,
  ClientInputMessageContent,
  FunctionConfig,
  JsonValue,
  PathWithContents,
  UninitializedVariantInfo,
  VariantInfo,
  Tool,
} from "tensorzero-node";
import type {
  InputMessageContent as TensorZeroContent,
  ImageContent as TensorZeroImage,
  InputMessage as TensorZeroMessage,
  Input as TensorZeroInput,
} from "~/utils/tensorzero";
import type {
  ResolvedFileContent,
  DisplayInputMessageContent,
  DisplayInput,
} from "~/utils/clickhouse/common";
import type { InferenceUsage } from "~/utils/clickhouse/helpers";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";
import type { ParsedDatasetRow } from "~/utils/clickhouse/datasets";
import type { InferenceResponse } from "~/utils/tensorzero";
import { logger } from "~/utils/logger";
import type {
  ClientInferenceParams,
  ResolvedInput as TensorZeroResolvedInput,
  ResolvedInputMessage as TensorZeroResolvedInputMessage,
  ResolvedInputMessageContent as TensorZeroResolvedInputMessageContent,
  ToolCallConfigDatabaseInsert,
  ContentBlockChatOutput,
  JsonInferenceOutput,
} from "tensorzero-node";
import type {
  Input,
  InputMessage,
  InputMessageContent,
} from "~/utils/clickhouse/common";

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
            info: {
              output:
                "content" in inferenceOutput
                  ? inferenceOutput.content
                  : inferenceOutput.output,
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

// Convert TensorZero's ResolvedInput to our Input type
export function tensorZeroResolvedInputToInput(
  resolvedInput: TensorZeroResolvedInput,
): Input {
  return {
    system: resolvedInput.system ?? undefined,
    messages: resolvedInput.messages.map(
      tensorZeroResolvedMessageToInputMessage,
    ),
  };
}

function tensorZeroResolvedMessageToInputMessage(
  message: TensorZeroResolvedInputMessage,
): InputMessage {
  return {
    role: message.role,
    content: message.content.map(tensorZeroResolvedContentToInputContent),
  };
}

function tensorZeroResolvedContentToInputContent(
  content: TensorZeroResolvedInputMessageContent,
): InputMessageContent {
  switch (content.type) {
    case "text":
      return content;
    case "tool_call":
      return {
        type: "tool_call",
        id: content.id,
        name: content.name,
        arguments: content.arguments,
      };
    case "tool_result":
      return {
        type: "tool_result",
        id: content.id,
        name: content.name,
        result: content.result,
      };
    case "raw_text":
      return content;
    case "thought":
      return {
        type: "thought",
        text: content.text,
        _internal_provider_type: content._internal_provider_type,
        signature: content.signature,
      };
    case "file": {
      // Handle the StoragePath conversion properly
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
          url: null,
          mime_type: content.file.mime_type,
        },
        storage_path: {
          path: content.storage_path.path,
          kind: convertedKind,
        },
      };
    }
    case "unknown":
      return {
        type: "unknown",
        data: content.data,
        model_provider_name: content.model_provider_name,
      };
  }
}

interface InferenceActionArgs {
  source: "inference";
  resource: ParsedInferenceRow;
  variant: string;
}

interface DatapointActionArgs {
  source: "datapoint";
  resource: ParsedDatasetRow;
  variant: string;
}

interface ClickHouseDatapointActionArgs {
  source: "clickhouse_datapoint";
  input: DisplayInput;
  functionName: string;
  // Optional fields for json / chat datapoints
  tool_params?: ToolCallConfigDatabaseInsert;
  output_schema?: JsonValue;
  variant?: string;
  cache_options: CacheParamsOptions;
  dryrun: boolean;
  editedVariantInfo?: VariantInfo;
  functionConfig: FunctionConfig;
}

export function prepareInferenceActionRequest(
  args:
    | InferenceActionArgs
    | DatapointActionArgs
    | ClickHouseDatapointActionArgs,
): ClientInferenceParams {
  // Create base ClientInferenceParams with default values
  const baseParams: ClientInferenceParams = {
    function_name: null,
    model_name: null,
    episode_id: null,
    input: { system: null, messages: [] },
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
    dryrun: null,
    internal: false,
    tags: {},
    output_schema: null,
    credentials: new Map(),
    cache_options: {
      max_age_s: null,
      enabled: "on",
    },
    include_original_response: false,
    internal_dynamic_variant_config: null,
    allowed_tools: null,
    additional_tools: null,
    tool_choice: null,
    parallel_tool_calls: null,
  };

  // Prepare request based on source and function type
  if (
    args.source === "inference" &&
    args.resource.function_name === "tensorzero::default"
  ) {
    const defaultRequest = prepareDefaultFunctionRequest(
      args.resource,
      args.variant,
    );
    return { ...baseParams, ...defaultRequest };
  } else if (args.source === "clickhouse_datapoint") {
    // Extract tool parameters from the ClickHouse datapoint args
    const tool_choice = args.tool_params?.tool_choice;
    const parallel_tool_calls = args.tool_params?.parallel_tool_calls;
    const dynamicVariantInfo = args.editedVariantInfo
      ? variantInfoToUninitalizedVariantInfo(args.editedVariantInfo)
      : null;
    const additional_tools = args.tool_params?.tools_available
      ? subtractStaticToolsFromInferenceInput(
          args.tool_params?.tools_available,
          args.functionConfig,
        )
      : null;

    return {
      ...baseParams,
      function_name: args.functionName,
      input: resolvedInputToClientInput(args.input),
      variant_name: args.variant || null,
      output_schema: args.output_schema || null,
      tool_choice: tool_choice || null,
      dryrun: args.dryrun,
      parallel_tool_calls: parallel_tool_calls || null,
      additional_tools,
      cache_options: args.cache_options,
      internal_dynamic_variant_config: dynamicVariantInfo,
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
    const clientInput = resolvedInputToClientInput(args.resource.input);
    // TODO: this is unsupported in Node bindings for now
    // const extra_body =
    //   args.source === "inference" ? args.resource.extra_body : undefined;

    return {
      ...baseParams,
      function_name: args.resource.function_name,
      input: clientInput,
      variant_name: args.variant,
      dryrun: true,
    };
  }
}

function prepareDefaultFunctionRequest(
  inference: ParsedInferenceRow,
  selectedVariant: string,
): Partial<ClientInferenceParams> {
  const clientInput = resolvedInputToClientInput(inference.input);
  if (inference.function_type === "chat") {
    const tool_choice = inference.tool_params?.tool_choice;
    const parallel_tool_calls = inference.tool_params?.parallel_tool_calls;
    const tools_available = inference.tool_params?.tools_available;
    return {
      model_name: selectedVariant,
      input: clientInput,
      dryrun: true,
      tool_choice: tool_choice,
      parallel_tool_calls: parallel_tool_calls,
      // We need to add all tools as additional for the default function
      additional_tools: tools_available,
    };
  } else if (inference.function_type === "json") {
    // This should never happen, just in case and for type safety
    const output_schema = inference.output_schema;
    return {
      model_name: selectedVariant,
      input: clientInput,
      dryrun: true,
      output_schema: output_schema || null,
    };
  }

  // Fallback case
  return {
    model_name: selectedVariant,
    input: clientInput,
    dryrun: true,
  };
}

export interface VariantResponseInfo {
  output?: JsonInferenceOutput | ContentBlockChatOutput[];
  usage?: InferenceUsage;
}

export function resolvedInputToClientInput(input: DisplayInput): ClientInput {
  return {
    system: input.system || null,
    messages: input.messages.map(resolvedInputMessageToClientInputMessage),
  };
}

export function resolvedInputToTensorZeroInput(
  input: DisplayInput,
): TensorZeroInput {
  return {
    ...input,
    messages: input.messages.map(resolvedInputMessageToTensorZeroMessage),
  };
}

function resolvedInputMessageToTensorZeroMessage(
  message: DisplayInputMessage,
): TensorZeroMessage {
  return {
    ...message,
    content: message.content.map(
      resolvedInputMessageContentToTensorZeroContent,
    ),
  };
}

function resolvedInputMessageContentToTensorZeroContent(
  content: DisplayInputMessageContent,
): TensorZeroContent {
  switch (content.type) {
    case "structured_text":
      return {
        type: "text",
        arguments: content.arguments,
      };
    case "unstructured_text":
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
    case "unknown":
      return content;
    case "file":
      return resolvedFileContentToTensorZeroFile(content);
    case "file_error":
      throw new Error("Can't convert image error to tensorzero content");
  }
}

function resolvedFileContentToTensorZeroFile(
  content: ResolvedFileContent,
): TensorZeroImage {
  const data = content.file.dataUrl.split(",")[1];
  return {
    type: "image",
    mime_type: content.file.mime_type,
    data,
  };
}

function resolvedInputMessageToClientInputMessage(
  message: DisplayInputMessage,
): ClientInputMessage {
  return {
    role: message.role,
    content: message.content.map(
      resolvedInputMessageContentToClientInputMessageContent,
    ),
  };
}

function resolvedInputMessageContentToClientInputMessageContent(
  content: DisplayInputMessageContent,
): ClientInputMessageContent {
  switch (content.type) {
    case "structured_text":
      return {
        type: "text",
        arguments: content.arguments,
      };
    case "unstructured_text":
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
        signature: content.signature || null,
        _internal_provider_type: null,
      };
    case "unknown":
      return {
        type: "unknown",
        data: content.data,
        model_provider_name: content.model_provider_name || null,
      };
    case "file":
      return resolvedFileContentToClientFile(content);
    case "file_error":
      throw new Error("Can't convert image error to client content");
  }
}

function resolvedFileContentToClientFile(
  content: ResolvedFileContent,
): ClientInputMessageContent {
  const data = content.file.dataUrl.split(",")[1];
  return {
    type: "file",
    mime_type: content.file.mime_type,
    data: data,
  };
}

function variantInfoToUninitalizedVariantInfo(
  variantInfo: VariantInfo,
): UninitializedVariantInfo {
  const convertTemplate = (template: PathWithContents | null) => {
    if (!template) return null;
    return {
      __tensorzero_remapped_path: `template_${Math.random().toString(36).substring(2, 15)}`,
      __data: template.contents,
    };
  };
  const stringToTemplate = (template: string | null) => {
    if (!template) return null;
    return {
      __tensorzero_remapped_path: `template_${Math.random().toString(36).substring(2, 15)}`,
      __data: template,
    };
  };

  const baseUninitialized = {
    timeouts: variantInfo.timeouts,
  };

  const inner = variantInfo.inner;

  switch (inner.type) {
    case "chat_completion":
      return {
        ...baseUninitialized,
        type: "chat_completion" as const,
        weight: inner.weight,
        model: inner.model,
        input_wrappers: null,
        system_template: convertTemplate(
          inner.templates.system?.template || null,
        ),
        user_template: convertTemplate(inner.templates.user?.template || null),
        assistant_template: convertTemplate(
          inner.templates.assistant?.template || null,
        ),
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

    case "best_of_n_sampling":
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
          system_template: convertTemplate(
            inner.evaluator.templates.system?.template || null,
          ),
          user_template: convertTemplate(
            inner.evaluator.templates.user?.template || null,
          ),
          assistant_template: convertTemplate(
            inner.evaluator.templates.assistant?.template || null,
          ),
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

    case "dicl":
      return {
        ...baseUninitialized,
        type: "experimental_dynamic_in_context_learning" as const,
        weight: inner.weight,
        embedding_model: inner.embedding_model,
        k: inner.k,
        model: inner.model,
        system_instructions: stringToTemplate(inner.system_instructions),
        temperature: inner.temperature,
        top_p: inner.top_p,
        stop_sequences: inner.stop_sequences,
        presence_penalty: inner.presence_penalty,
        frequency_penalty: inner.frequency_penalty,
        max_tokens: inner.max_tokens,
        seed: inner.seed,
        json_mode: inner.json_mode,
        retries: inner.retries,
      };

    case "mixture_of_n":
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
          system_template: convertTemplate(
            inner.fuser.templates.system?.template || null,
          ),
          user_template: convertTemplate(
            inner.fuser.templates.user?.template || null,
          ),
          assistant_template: convertTemplate(
            inner.fuser.templates.assistant?.template || null,
          ),
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

    case "chain_of_thought":
      return {
        ...baseUninitialized,
        type: "experimental_chain_of_thought" as const,
        weight: inner.weight,
        model: inner.model,
        input_wrappers: null,
        system_template: convertTemplate(
          inner.templates.system?.template || null,
        ),
        user_template: convertTemplate(inner.templates.user?.template || null),
        assistant_template: convertTemplate(
          inner.templates.assistant?.template || null,
        ),
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

    default:
      throw new Error(`Unknown variant type`);
  }
}

/*
 * For both inferences and datapoints, we store a full tool config that
 * specifies what the model saw or could have seen at inference time for a particular example.
 * However, TensorZero will automatically use the tools that are currently configured for inferences.
 * It will also error if there are tools with duplicated names. In order to avoid this, we "subtract"
 * out all currently configured tools from the tools that we pass in dynamically.
 */
function subtractStaticToolsFromInferenceInput(
  datapointTools: Tool[],
  functionConfig: FunctionConfig,
): Tool[] {
  if (functionConfig.type === "json") {
    return datapointTools;
  }
  const resultTools = [];
  for (const tool of datapointTools) {
    if (!functionConfig.tools.some((t) => t === tool.name)) {
      resultTools.push(tool);
    }
  }
  return resultTools;
}
