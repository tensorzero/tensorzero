import { Loader2 } from "lucide-react";
import { Suspense } from "react";
import { Await, useAsyncError } from "react-router";
import { Refresh } from "~/components/icons/Icons";
import NewOutput from "~/components/inference/NewOutput";
import { Button } from "~/components/ui/button";
import { CodeEditor } from "~/components/ui/code-editor";
import { refreshClientInference } from "./utils";
import type { InferenceResponse } from "~/utils/tensorzero";
import type { Datapoint as TensorZeroDatapoint } from "tensorzero-node";
import type { DisplayInput } from "~/utils/clickhouse/common";

interface DatapointPlaygroundOutputProps {
  datapoint: TensorZeroDatapoint;
  variantName: string;
  serverInference: Promise<InferenceResponse> | undefined;
  setPromise: (
    variantName: string,
    datapointId: string,
    promise: Promise<InferenceResponse>,
  ) => void;
  input: DisplayInput;
  functionName: string;
}
export default function DatapointPlaygroundOutput({
  datapoint,
  variantName,
  serverInference,
  setPromise,
  input,
  functionName,
}: DatapointPlaygroundOutputProps) {
  if (!serverInference) {
    return (
      <div className="flex min-h-[8rem] items-center justify-center">
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-1 left-1 z-10 opacity-0 transition-opacity group-hover:opacity-100"
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
        <div className="text-muted-foreground text-sm">
          No inference available
        </div>
      </div>
    );
  }

  return (
    <div className="group relative">
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-1 left-1 z-10 opacity-0 transition-opacity group-hover:opacity-100"
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
      <Suspense
        fallback={
          <div className="flex min-h-[8rem] items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin" />
          </div>
        }
      >
        <Await resolve={serverInference} errorElement={<InferenceError />}>
          {(response) => {
            if (!response) {
              return (
                <div className="flex min-h-[8rem] items-center justify-center">
                  <div className="text-muted-foreground text-sm">
                    No response available
                  </div>
                </div>
              );
            }
            console.log(response);
            let output;
            if ("content" in response) {
              output = response.content;
            } else {
              output = response.output;
            }
            return <NewOutput output={output} />;
          }}
        </Await>
      </Suspense>
    </div>
  );
}

function InferenceError() {
  const error = useAsyncError();
  const isInferenceError = error instanceof Error;

  return (
    <div className="flex min-h-[8rem] items-center justify-center">
      <div className="max-h-[16rem] max-w-md overflow-y-auto px-4 text-center text-red-600">
        <p className="font-semibold">Error</p>
        <p className="mt-1 text-sm">
          {isInferenceError ? (
            <CodeEditor value={error.message} readOnly />
          ) : (
            "Failed to load inference"
          )}
        </p>
      </div>
    </div>
  );
}
