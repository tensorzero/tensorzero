import { Loader2, RefreshCw, AlertCircle } from "lucide-react";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";
import { Button } from "~/components/ui/button";
import { CodeEditor } from "~/components/ui/code-editor";
import {
  type ClientInferenceInputArgs,
  getClientInferenceQueryKey,
  getClientInferenceQueryFunction,
} from "./utils";
import { variantInfoToUninitializedVariantInfo } from "~/routes/api/tensorzero/inference.utils";
import { useQuery } from "@tanstack/react-query";
import { isErrorLike } from "~/utils/common";
import { memo, useState, useEffect, useRef } from "react";
import { Link } from "react-router";
import { toInferenceUrl } from "~/utils/urls";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "~/components/ui/tooltip";
import { MetricBadge } from "~/components/metric/MetricBadge";
import { AnimatedEllipsis } from "~/components/ui/AnimatedEllipsis";
import type { JsonValue } from "~/types/tensorzero";

type EvaluationResult = {
  evaluations: Record<string, JsonValue | null | undefined>;
  evaluatorErrors: Record<string, string>;
};

type EvaluationState = {
  status: "idle" | "loading" | "success" | "error";
  result?: EvaluationResult;
  error?: string;
};

type EvaluationEvent = {
  type: "success" | "error" | "start" | "complete" | "fatal_error";
  datapoint?: { id: string };
  evaluations?: Record<string, JsonValue | null | undefined>;
  evaluator_errors?: Record<string, string>;
  error?: string;
  message?: string;
};

type DatapointPlaygroundOutputProps = ClientInferenceInputArgs & {
  selectedEvaluation: string | null;
};

const DatapointPlaygroundOutput = memo<DatapointPlaygroundOutputProps>(
  function DatapointPlaygroundOutput(props) {
    const { selectedEvaluation, datapoint, variant } = props;
    const [evalState, setEvalState] = useState<EvaluationState>({
      status: "idle",
    });
    const abortControllerRef = useRef<AbortController | null>(null);

    const query = useQuery({
      queryKey: getClientInferenceQueryKey(props),
      queryFn: getClientInferenceQueryFunction(props),
      refetchOnMount: false,
      refetchOnWindowFocus: false,
      refetchInterval: false,
      retry: false,
    });

    // For edited variants, track config changes to re-run evaluations
    const variantConfig = variant.type === "edited" ? variant.config : null;

    // Run evaluation when inference is ready and evaluation is selected
    useEffect(() => {
      if (!selectedEvaluation || !query.isSuccess || query.isRefetching) {
        if (!selectedEvaluation) {
          setEvalState({ status: "idle" });
        }
        return;
      }

      // Abort any previous request
      abortControllerRef.current?.abort();
      abortControllerRef.current = new AbortController();

      setEvalState({ status: "loading" });

      const runEval = async () => {
        try {
          // For builtin variants, pass variantName; for edited variants, pass variantConfig
          const variantPayload =
            variant.type === "builtin" || !variantConfig
              ? { variantName: variant.name }
              : {
                  variantConfig: JSON.stringify(
                    variantInfoToUninitializedVariantInfo(variantConfig),
                  ),
                };

          const response = await fetch("/api/playground/evaluate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              evaluationName: selectedEvaluation,
              ...variantPayload,
              datapointIds: [datapoint.id],
            }),
            signal: abortControllerRef.current?.signal,
          });

          if (!response.ok) {
            const errorText = await response.text();
            setEvalState({ status: "error", error: errorText });
            return;
          }

          const reader = response.body?.getReader();
          if (!reader) {
            setEvalState({ status: "error", error: "No response body" });
            return;
          }

          const decoder = new TextDecoder();
          let buffer = "";

          const processLine = (line: string) => {
            if (!line.trim()) return;
            try {
              const event = JSON.parse(line) as EvaluationEvent;
              if (
                event.type === "success" &&
                event.datapoint?.id === datapoint.id
              ) {
                setEvalState({
                  status: "success",
                  result: {
                    evaluations: event.evaluations ?? {},
                    evaluatorErrors: event.evaluator_errors ?? {},
                  },
                });
              } else if (
                event.type === "error" &&
                event.datapoint?.id === datapoint.id
              ) {
                setEvalState({
                  status: "error",
                  error: event.error || "Evaluation failed",
                });
              } else if (event.type === "fatal_error") {
                setEvalState({
                  status: "error",
                  error: event.message || "Evaluation failed",
                });
              } else if (event.type === "complete") {
                // If still loading after stream completes, no result was returned for this datapoint
                setEvalState((prev) =>
                  prev.status === "loading"
                    ? {
                        status: "error",
                        error: "No evaluation result received",
                      }
                    : prev,
                );
              }
            } catch {
              // Ignore parse errors for malformed lines
            }
          };

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              processLine(line);
            }
          }

          // Process any remaining data in buffer after stream ends
          if (buffer.trim()) {
            processLine(buffer);
          }
        } catch (err) {
          if (err instanceof Error && err.name === "AbortError") {
            return;
          }
          setEvalState({
            status: "error",
            error: err instanceof Error ? err.message : "Unknown error",
          });
        }
      };

      runEval();

      return () => {
        abortControllerRef.current?.abort();
      };
    }, [
      selectedEvaluation,
      variant.type,
      variant.name,
      variantConfig,
      datapoint.id,
      query.isSuccess,
      query.isRefetching,
    ]);

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
        {evalState.status !== "idle" && selectedEvaluation && (
          <div className="mt-2">
            {evalState.status === "loading" && (
              <div className="text-muted-foreground flex items-center gap-1.5 text-xs">
                <Loader2 className="h-3 w-3 animate-spin" />
                <span>
                  Evaluating
                  <AnimatedEllipsis />
                </span>
              </div>
            )}
            {evalState.status === "error" && (
              <div className="flex items-center gap-1.5 text-xs text-red-600">
                <AlertCircle className="h-3 w-3" />
                <span>{evalState.error || "Evaluation failed"}</span>
              </div>
            )}
            {evalState.status === "success" &&
              evalState.result?.evaluations && (
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(evalState.result.evaluations).map(
                    ([evaluatorName, value]) => (
                      <MetricBadge
                        key={evaluatorName}
                        label={evaluatorName}
                        value={value}
                        error={Boolean(
                          evalState.result?.evaluatorErrors?.[evaluatorName],
                        )}
                      />
                    ),
                  )}
                </div>
              )}
          </div>
        )}
      </div>
    );
  },
  (prevProps, nextProps) => {
    return (
      prevProps.datapoint.id === nextProps.datapoint.id &&
      prevProps.variant.name === nextProps.variant.name &&
      prevProps.functionName === nextProps.functionName &&
      prevProps.selectedEvaluation === nextProps.selectedEvaluation &&
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
