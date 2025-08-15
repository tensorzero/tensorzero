import { Loader2 } from "lucide-react";
import { Refresh } from "~/components/icons/Icons";
import { Output } from "~/components/inference/Output";
import { Button } from "~/components/ui/button";
import { CodeEditor } from "~/components/ui/code-editor";
import {
  preparePlaygroundInferenceRequest,
  fetchClientInference,
  type PlaygroundVariantInfo,
} from "./utils";
import type { DisplayInput } from "~/utils/clickhouse/common";
import type { Datapoint, FunctionConfig } from "tensorzero-node";
import { useQuery } from "@tanstack/react-query";
import { isErrorLike } from "~/utils/common";
import { memo } from "react";

interface DatapointPlaygroundOutputProps {
  datapoint: Datapoint;
  variant: PlaygroundVariantInfo;
  input: DisplayInput;
  functionName: string;
  functionConfig: FunctionConfig;
}

const DatapointPlaygroundOutput = memo(function DatapointPlaygroundOutput({
  datapoint,
  variant,
  input,
  functionName,
  functionConfig,
}: DatapointPlaygroundOutputProps) {
  const query = useQuery({
    queryKey: ["DATASETS_COUNT", { variant, datapoint, input, functionConfig }],
    queryFn: async ({ signal }) => {
      return await fetchClientInference(
        preparePlaygroundInferenceRequest(
          variant,
          functionName,
          datapoint,
          input,
          functionConfig,
        ),
        { signal },
      );
    },
    refetchOnMount: false,
  });

  const loadingIndicator = (
    <div className="flex min-h-[8rem] items-center justify-center">
      <Loader2 className="h-8 w-8 animate-spin" aria-hidden />
    </div>
  );

  const refreshButton = (
    <Button
      aria-label={`Reload ${variant.name} inference`}
      variant="ghost"
      size="icon"
      className="absolute top-1 right-1 z-5 cursor-pointer opacity-25 transition-opacity hover:opacity-100"
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

  return (
    <div className="group relative">
      {refreshButton}
      <Output output={output} maxHeight={480} />
    </div>
  );
});

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
