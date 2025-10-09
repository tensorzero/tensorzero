import { Loader2 } from "lucide-react";
import { Refresh } from "~/components/icons/Icons";
import { Output } from "~/components/inference/Output";
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
        size="icon"
        className="absolute top-1 right-1 z-5 cursor-pointer opacity-25 transition-opacity hover:opacity-100"
        data-testid="datapoint-playground-output-refresh-button"
        onClick={() => query.refetch()}
      >
        <Refresh />
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

    const output =
      "content" in query.data ? query.data.content : query.data.output;
    const inferenceId = query.data.inference_id;

    return (
      <div className="group relative" data-testid="datapoint-playground-output">
        {refreshButton}
        <Output output={output} maxHeight={480} />
        {inferenceId && (
          <div className="mt-2 text-xs">
            Inference ID:{" "}
            <Link
              to={toInferenceUrl(inferenceId)}
              className="font-mono text-xs text-blue-600 hover:text-blue-800 hover:underline"
            >
              {inferenceId}
            </Link>
          </div>
        )}
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
