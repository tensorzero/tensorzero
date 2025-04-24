import * as React from "react";
import { useFetcher } from "react-router";
import type {
  SubmitTarget,
  FetcherSubmitOptions,
  FetcherWithComponents,
} from "react-router";
import type {
  ContentBlockOutput,
  JsonInferenceOutput,
} from "~/utils/clickhouse/common";
import type { InferenceUsage } from "~/utils/clickhouse/helpers";
import type { InferenceResponse } from "~/utils/tensorzero";
import { resolvedInputToTensorZeroInput } from "./inference";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";
import type { ParsedDatasetRow } from "~/utils/clickhouse/datasets";

interface InferenceActionError {
  message: string;
  caught: unknown;
}

type InferenceActionResponse = {
  info: VariantResponseInfo;
  raw: InferenceResponse;
};

type InferenceActionContext =
  | { state: "init"; data: null; error: null }
  | { state: "idle"; data: InferenceActionResponse | null; error: null }
  | { state: "submitting"; data: InferenceActionResponse | null; error: null }
  | { state: "loading"; data: InferenceActionResponse | null; error: null }
  | {
      state: "error";
      data: (Pick<InferenceActionResponse, "raw"> & { info?: never }) | null;
      error: InferenceActionError;
    };

const ENDPOINT = "/api/tensorzero/inference";

type ActionFetcher = InferenceActionContext & {
  submit(
    target: SubmitTarget,
    opts?: Omit<FetcherSubmitOptions, "method" | "method" | "action">,
  ): Promise<void>;
  _raw: FetcherWithComponents<InferenceResponse>;
};

export function useInferenceActionFetcher() {
  const fetcher = useFetcher<InferenceResponse>();
  const context = React.useMemo<InferenceActionContext>(() => {
    if (fetcher.data) {
      try {
        const inferenceOutput = fetcher.data;
        // Check if the response contains an error
        if ("error" in inferenceOutput && inferenceOutput.error) {
          return {
            state: "error",
            data: { raw: fetcher.data },
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
          data: { raw: fetcher.data },
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

  const { submit: fetcherSubmit } = fetcher;
  const submit = React.useCallback<ActionFetcher["submit"]>(
    (target, opts) => {
      return fetcherSubmit(target, {
        ...opts,
        method: "POST",
        action: ENDPOINT,
      });
    },
    [fetcherSubmit],
  );

  React.useEffect(() => {
    if (context.error?.caught) {
      console.error("Error processing response:", context.error.caught);
    }
  }, [context.error]);

  return {
    ...context,
    submit,
    // shouldn't need this, used as an escape hatch
    _raw: fetcher,
  } satisfies ActionFetcher;
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

export function prepareInferenceActionRequest(
  args: InferenceActionArgs | DatapointActionArgs,
) {
  // Prepare request based on source and function type
  let request;
  if (
    args.source === "inference" &&
    args.resource.function_name === "tensorzero::default"
  ) {
    request = prepareDefaultFunctionRequest(args.resource, args.variant);
  } else {
    const tensorZeroInput = resolvedInputToTensorZeroInput(args.resource.input);
    request = {
      function_name: args.resource.function_name,
      input: tensorZeroInput,
      variant_name: args.variant,
      dryrun: true,
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
