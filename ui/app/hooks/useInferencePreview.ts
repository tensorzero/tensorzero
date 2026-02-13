import { useEffect } from "react";
import { useFetcher } from "react-router";

export interface InferencePreview {
  inference_id: string;
  function_name: string;
  variant_name: string;
  episode_id: string;
  timestamp: string;
  processing_time_ms: number | null;
  type: "chat" | "json";
}

export function useInferencePreview(
  inferenceId: string,
  enabled: boolean,
): { data: InferencePreview | null; isLoading: boolean } {
  const fetcher = useFetcher<InferencePreview>({
    key: `inference-preview-${inferenceId}`,
  });

  useEffect(() => {
    if (enabled && fetcher.state === "idle" && !fetcher.data) {
      fetcher.load(
        `/api/tensorzero/inference_preview/${encodeURIComponent(inferenceId)}`,
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inferenceId, enabled]);

  return {
    data: fetcher.data ?? null,
    isLoading: fetcher.state !== "idle",
  };
}
