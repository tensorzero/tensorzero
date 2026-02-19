import { useCallback, useEffect, useRef, useState } from "react";
import { useFetcher } from "react-router";
import { useToast } from "~/hooks/use-toast";

interface CancelEvaluationResponse {
  success: boolean;
  error?: string;
}

export function useCancelEvaluation({
  runningEvaluationRunIds,
}: {
  runningEvaluationRunIds: string[];
}) {
  const { toast } = useToast();
  const fetcher = useFetcher<CancelEvaluationResponse>();
  const [isCancelling, setIsCancelling] = useState(false);
  const hasSubmittedRef = useRef(false);

  const anyEvaluationIsRunning =
    runningEvaluationRunIds.length > 0 && !isCancelling;

  // Reset cancel state once server confirms no more running evaluations
  useEffect(() => {
    if (isCancelling && runningEvaluationRunIds.length === 0) {
      setIsCancelling(false);
    }
  }, [isCancelling, runningEvaluationRunIds]);

  // Handle fetcher response (success, error response, or network error)
  useEffect(() => {
    if (fetcher.state !== "idle") {
      hasSubmittedRef.current = true;
      return;
    }
    if (!hasSubmittedRef.current) return;
    hasSubmittedRef.current = false;

    if (fetcher.data?.success) {
      const { dismiss } = toast.success({ title: "Evaluation stopped" });
      return () => dismiss({ immediate: true });
    }

    // Error response from server
    if (fetcher.data?.error) {
      toast.error({
        title: "Failed to stop evaluation",
        description: fetcher.data.error,
      });
    } else {
      // Network error — fetcher completed but no data returned
      toast.error({ title: "Failed to stop evaluation" });
    }
    setIsCancelling(false);
    return;
  }, [fetcher.state, fetcher.data, toast]);

  const handleCancelEvaluation = useCallback(() => {
    setIsCancelling(true);
    fetcher.submit(
      { evaluation_run_ids: runningEvaluationRunIds },
      {
        method: "POST",
        action: "/api/evaluations/cancel",
        encType: "application/json",
      },
    );
  }, [runningEvaluationRunIds, fetcher]);

  return {
    isCancelling,
    anyEvaluationIsRunning,
    handleCancelEvaluation,
  };
}
