import { Loader2, RefreshCw } from "lucide-react";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";
import { Button } from "~/components/ui/button";
import { CodeEditor } from "~/components/ui/code-editor";
import {
  type ClientInferenceInputArgs,
  getClientInferenceQueryKey,
  getClientInferenceQueryFunction,
} from "./utils";
import { useQuery } from "@tanstack/react-query";
import { isErrorLike } from "~/utils/common";
import { memo } from "react";
import { Link } from "react-router";
import { toInferenceUrl } from "~/utils/urls";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "~/components/ui/tooltip";

const DatapointPlaygroundOutput = memo<ClientInferenceInputArgs>(
  function DatapointPlaygroundOutput(props) {
    const query = useQuery({
      queryKey: getClientInferenceQueryKey(props),
      queryFn: getClientInferenceQueryFunction(props),
      // Only re-fetch when the user explicitly requests it
      refetchOnMount: false,
      refetchInterval: false,
      retry: false,
    });

    const loadingIndicator = (
      <div
        className="flex min-h-[8rem] items-center justify-center"
        data-testid="datapoint-playground-output-loading"
      >
        <Loader2 className="h-8 w-8 animate-spin" aria-hidden />
      </div>
    );

    const refreshButton = (
      <Button
        aria-label={`Reload ${props.variant.name} inference`}
        variant="ghost"
        size="iconSm"
        className="absolute top-1 right-1 z-5 h-6 w-6 cursor-pointer text-xs opacity-25 transition-opacity hover:opacity-100"
        data-testid="datapoint-playground-output-refresh-button"
        onClick={() => query.refetch()}
      >
        <RefreshCw />
      </Button>
    );

    if (query.isLoading || query.isRefetching) {
      return loadingIndicator;
    }

    if (query.isError) {
      return (
        <>
          {refreshButton}
          <InferenceError error={query.error} />
        </>
      );
    }

    if (!query.data) {
      return (
        <div className="flex min-h-[8rem] items-center justify-center">
          {refreshButton}
          <div className="text-muted-foreground text-sm">
            No inference available
          </div>
        </div>
      );
    }

    return (
      <div
        className="flex flex-col gap-2"
        data-testid="datapoint-playground-output"
      >
        <div className="relative">
          {refreshButton}
          {props.variant.type === "builtin" && (
            <div className="mt-2 text-xs">
              Inference ID:{" "}
              <Link
                to={toInferenceUrl(query.data.inference_id)}
                className="font-mono text-xs text-blue-600 hover:text-blue-800 hover:underline"
              >
                {query.data.inference_id}
              </Link>
            </div>
          )}
          {props.variant.type === "edited" && (
            <div className="mt-2 text-xs">
              Inference ID:{" "}
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-muted-foreground cursor-help underline decoration-dotted">
                    none
                  </span>
                </TooltipTrigger>
                <TooltipContent side="top">
                  <p className="text-xs">
                    Edited variants currently run with{" "}
                    <span className="font-mono text-xs">dryrun</span> set to{" "}
                    <span className="font-mono text-xs">true</span>, so the
                    inference was not stored.
                  </p>
                </TooltipContent>
              </Tooltip>
            </div>
          )}
        </div>
        <div>
          {"content" in query.data ? (
            <ChatOutputElement output={query.data.content} maxHeight={480} />
          ) : (
            <JsonOutputElement output={query.data.output} maxHeight={480} />
          )}
        </div>
      </div>
    );
  },
  // TODO: Remove custom comparison and make props stable instead
  (prevProps, nextProps) => {
    return (
      prevProps.datapoint.id === nextProps.datapoint.id &&
      prevProps.variant.name === nextProps.variant.name &&
      prevProps.functionName === nextProps.functionName &&
      JSON.stringify(prevProps.input) === JSON.stringify(nextProps.input)
    );
  },
);

export default DatapointPlaygroundOutput;

function InferenceError({ error }: { error: unknown }) {
  return (
    <div className="max-h-[16rem] max-w-md overflow-y-auto px-4 text-red-600">
      <h3 className="text-sm font-medium">Inference Error</h3>
      <div className="mt-2 text-sm">
        {isErrorLike(error) ? (
          <CodeEditor value={error.message} readOnly showLineNumbers={false} />
        ) : (
          "Failed to load inference"
        )}
      </div>
    </div>
  );
}
