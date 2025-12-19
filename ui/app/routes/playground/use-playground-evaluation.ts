import { useState, useEffect, useRef } from "react";
import { variantInfoToUninitializedVariantInfo } from "~/routes/api/tensorzero/inference.utils";
import type { JsonValue } from "~/types/tensorzero";
import type { PlaygroundVariantInfo } from "./utils";

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

type UsePlaygroundEvaluationParams = {
  selectedEvaluation: string | null;
  datapointId: string;
  variant: PlaygroundVariantInfo;
  /** Whether the inference query has succeeded (ready for evaluation) */
  isInferenceReady: boolean;
  /** Whether the inference is currently refetching */
  isInferenceRefetching: boolean;
};

/**
 * Hook to run playground evaluations after inference completes.
 * Streams evaluation results from the /api/playground/evaluate endpoint.
 */
export function usePlaygroundEvaluation({
  selectedEvaluation,
  datapointId,
  variant,
  isInferenceReady,
  isInferenceRefetching,
}: UsePlaygroundEvaluationParams): EvaluationState {
  const [evalState, setEvalState] = useState<EvaluationState>({
    status: "idle",
  });
  const abortControllerRef = useRef<AbortController | null>(null);

  // For edited variants, track config changes to re-run evaluations
  const variantConfig = variant.type === "edited" ? variant.config : null;

  useEffect(() => {
    if (!selectedEvaluation || !isInferenceReady || isInferenceRefetching) {
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
            datapointIds: [datapointId],
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
              event.datapoint?.id === datapointId
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
              event.datapoint?.id === datapointId
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
    datapointId,
    isInferenceReady,
    isInferenceRefetching,
  ]);

  return evalState;
}
