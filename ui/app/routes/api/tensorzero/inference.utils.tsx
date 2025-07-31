import * as React from "react";
import { useFetcher, type FetcherFormProps } from "react-router";
import type { SubmitTarget, FetcherSubmitOptions } from "react-router";
import type { DisplayInputMessage } from "~/utils/clickhouse/common";
import type {
  InputMessageContent as TensorZeroContent,
  ImageContent as TensorZeroImage,
  InputMessage as TensorZeroMessage,
  Input as TensorZeroInput,
} from "~/utils/tensorzero";
import type {
  ResolvedFileContent,
  DisplayInputMessageContent,
  ContentBlockOutput,
  DisplayInput,
  JsonInferenceOutput,
} from "~/utils/clickhouse/common";
import type { InferenceUsage } from "~/utils/clickhouse/helpers";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";
import type { ParsedDatasetRow } from "~/utils/clickhouse/datasets";
import type { InferenceResponse } from "~/utils/tensorzero";
import { logger } from "~/utils/logger";
import type {
  JsonValue,
  ResolvedInput as TensorZeroResolvedInput,
  ResolvedInputMessage as TensorZeroResolvedInputMessage,
  ResolvedInputMessageContent as TensorZeroResolvedInputMessageContent,
  ToolCallConfigDatabaseInsert,
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
  variant: string;
}

export function prepareInferenceActionRequest(
  args:
    | InferenceActionArgs
    | DatapointActionArgs
    | ClickHouseDatapointActionArgs,
) {
  // Prepare request based on source and function type
  let request;
  if (
    args.source === "inference" &&
    args.resource.function_name === "tensorzero::default"
  ) {
    request = prepareDefaultFunctionRequest(args.resource, args.variant);
  } else if (args.source === "clickhouse_datapoint") {
    // Convert TensorZero's ResolvedInput to our Input type first

    request = {
      function_name: args.functionName,
      input: resolvedInputToTensorZeroInput(args.input),
      variant_name: args.variant,
      tool_params: args.tool_params,
      output_schema: args.output_schema,
    };
  } else {
    // For other sources, the input is already a DisplayInput
    const tensorZeroInput = resolvedInputToTensorZeroInput(args.resource.input);
    const extra_body =
      args.source === "inference" ? args.resource.extra_body : undefined;
    request = {
      function_name: args.resource.function_name,
      input: tensorZeroInput,
      variant_name: args.variant,
      dryrun: true,
      extra_body,
    };
  }

  return request;
}

function prepareDefaultFunctionRequest(
  inference: ParsedInferenceRow,
  selectedVariant: string,
) {
  const tensorZeroInput = resolvedInputToTensorZeroInput(inference.input);
  if (inference.function_type === "chat") {
    const tool_choice = inference.tool_params?.tool_choice;
    const parallel_tool_calls = inference.tool_params?.parallel_tool_calls;
    const tools_available = inference.tool_params?.tools_available;
    return {
      model_name: selectedVariant,
      input: tensorZeroInput,
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
      input: tensorZeroInput,
      dryrun: true,
      output_schema: output_schema,
    };
  }
}

export interface VariantResponseInfo {
  output?: JsonInferenceOutput | ContentBlockOutput[];
  usage?: InferenceUsage;
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
