import { useState, useEffect, useRef } from "react";
import { useFetcher } from "react-router";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";

interface ActionData {
  inference?: ParsedInferenceRow;
  error?: string;
}

interface InferenceState {
  data: ParsedInferenceRow | null;
  loading: boolean;
  error: string | null;
}

export function useInferenceHover(episodeRoute: string) {
  const [hoveredInferenceId, setHoveredInferenceId] = useState<string | null>(null);
  const [inferenceCache, setInferenceCache] = useState<Record<string, InferenceState>>({});
  const fetcher = useFetcher<ActionData>();
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleInferenceHover = (inferenceId: string) => {
    setHoveredInferenceId(inferenceId);
    
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    const currentState = inferenceCache[inferenceId];
    if (currentState?.data || currentState?.loading) {
      return;
    }

    timeoutRef.current = setTimeout(() => {
      setInferenceCache(prev => ({
        ...prev,
        [inferenceId]: { data: null, loading: true, error: null }
      }));
      
      const formData = new FormData();
      formData.append("_action", "fetchInference");
      formData.append("inferenceId", inferenceId);
      
      fetcher.submit(formData, { 
        method: "POST",
        action: episodeRoute
      });
    }, 100);
  };

  useEffect(() => {
    if (fetcher.state === "idle" && fetcher.data && hoveredInferenceId) {
      setInferenceCache(prev => ({
        ...prev,
        [hoveredInferenceId]: {
          data: fetcher.data?.inference || null,
          loading: false,
          error: fetcher.data?.error || null
        }
      }));
    }
  }, [fetcher.state, fetcher.data, hoveredInferenceId]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return {
    handleInferenceHover,
    getInferenceData: (inferenceId: string) => inferenceCache[inferenceId]?.data || null,
    isLoading: (inferenceId: string) => inferenceCache[inferenceId]?.loading || false,
    getError: (inferenceId: string) => inferenceCache[inferenceId]?.error || null,
  };
}