import { Loader2 } from "lucide-react";
import { Suspense, memo } from "react";
import { Await, useAsyncError } from "react-router";
import { Refresh } from "~/components/icons/Icons";
import NewOutput from "~/components/inference/NewOutput";
import { Button } from "~/components/ui/button";
import { CodeEditor } from "~/components/ui/code-editor";
import { refreshClientInference } from "./utils";
import type { InferenceResponse } from "~/utils/tensorzero";
import type { DisplayInput } from "~/utils/clickhouse/common";
import type { Datapoint } from "tensorzero-node";

interface DatapointPlaygroundOutputProps {
  datapoint: Datapoint;
  variantName: string;
  serverInference: Promise<InferenceResponse> | undefined;
  isLoading?: boolean;
  setPromise: (
    variantName: string,
    datapointId: string,
    promise: Promise<InferenceResponse>,
  ) => void;
  input: DisplayInput;
  functionName: string;
}
const DatapointPlaygroundOutput = memo(
  function DatapointPlaygroundOutput({
    datapoint,
    variantName,
    serverInference,
    setPromise,
    input,
    functionName,
    isLoading,
  }: DatapointPlaygroundOutputProps) {
    const loadingIndicator = (
      <div className="flex min-h-[8rem] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin" aria-hidden />
      </div>
    );
    const refreshButton = (
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-1 right-1 z-5 cursor-pointer opacity-25 transition-opacity hover:opacity-100"
        onClick={() => {
          refreshClientInference(
            setPromise,
            input,
            datapoint,
            variantName,
            functionName,
          );
        }}
      >
        <Refresh />
      </Button>
    );

    if (!serverInference) {
      return isLoading ? (
        loadingIndicator
      ) : (
        <div className="flex min-h-[8rem] items-center justify-center">
          {refreshButton}
          <div className="text-muted-foreground text-sm">
            No inference available
          </div>
        </div>
      );
    }

    return (
      <div className="group relative">
        <Suspense fallback={loadingIndicator}>
          <Await
            resolve={serverInference}
            errorElement={
              <>
                {refreshButton}
                <InferenceError />
              </>
            }
          >
            {(response) => {
              if (!response) {
                return (
                  <>
                    {refreshButton}
                    <div className="flex min-h-[8rem] items-center justify-center">
                      <div className="text-muted-foreground text-sm">
                        No response available
                      </div>
                    </div>
                  </>
                );
              }
              let output;
              if ("content" in response) {
                output = response.content;
              } else {
                output = response.output;
              }
              return (
                <>
                  {refreshButton}
                  <NewOutput output={output} maxHeight={480} />
                </>
              );
            }}
          </Await>
        </Suspense>
      </div>
    );
  },
  (prevProps, nextProps) => {
    return (
      prevProps.datapoint.id === nextProps.datapoint.id &&
      prevProps.variantName === nextProps.variantName &&
      prevProps.functionName === nextProps.functionName &&
      prevProps.serverInference === nextProps.serverInference &&
      prevProps.setPromise === nextProps.setPromise &&
      JSON.stringify(prevProps.input) === JSON.stringify(nextProps.input)
    );
  },
);

export default DatapointPlaygroundOutput;

function InferenceError() {
  const error = useAsyncError();
  const isInferenceError = error instanceof Error;

  return (
    <div className="max-h-[16rem] max-w-md overflow-y-auto px-4 text-red-600">
      <h3 className="text-sm font-medium">Inference Error</h3>
      <div className="mt-2 text-sm">
        {isInferenceError ? (
          <CodeEditor value={error.message} readOnly showLineNumbers={false} />
        ) : (
          "Failed to load inference"
        )}
      </div>
    </div>
  );
}
