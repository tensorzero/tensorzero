import { useState, useEffect, useCallback } from "react";
import { useFetcher } from "react-router";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";

interface ActionData {
  inference?: ParsedInferenceRow;
  error?: string;
  inferenceId?: string;
}

interface InferenceState {
  data: ParsedInferenceRow | null;
  loading: boolean;
  error: string | null;
}

export function useInferenceClick(episodeRoute: string): {
  handleOpenSheet: (inferenceId: string) => void;
  handleCloseSheet: () => void;
  getInferenceData: (inferenceId: string) => ParsedInferenceRow | null;
  isLoading: (inferenceId: string) => boolean;
  getError: (inferenceId: string) => string | null;
  openSheetInferenceId: string | null;
} {
  const [openSheetInferenceId, setOpenSheetInferenceId] = useState<string | null>(null);
  const [inferenceCache, setInferenceCache] = useState<Record<string, InferenceState>>({});
  const fetcher = useFetcher<ActionData>();

  const handleOpenSheet = useCallback((inferenceId: string) => {
    setOpenSheetInferenceId(inferenceId);
    
    setInferenceCache(prev => {
      const currentState = prev[inferenceId];
      if (!currentState?.data && !currentState?.loading) {
        const formData = new FormData();
        formData.append("_action", "fetchInference");
        formData.append("inferenceId", inferenceId);
        
        fetcher.submit(formData, { 
          method: "POST",
          action: episodeRoute
        });
        
        return {
          ...prev,
          [inferenceId]: { data: null, loading: true, error: null }
        };
      }
      return prev;
    });
  }, [episodeRoute]);

  const handleCloseSheet = useCallback(() => {
    setOpenSheetInferenceId(null);
  }, []);

  useEffect(() => {
    if (fetcher.state !== "idle" || !fetcher.data) return;
    
    const { inferenceId, inference, error } = fetcher.data;
    
    if (inferenceId) {
      setInferenceCache(prev => ({
        ...prev,
        [inferenceId]: {
          data: inference || null,
          loading: false,
          error: error || null
        }
      }));
    }
  }, [fetcher.state, fetcher.data]);

  return {
    handleOpenSheet,
    handleCloseSheet,
    getInferenceData: (inferenceId: string) => inferenceCache[inferenceId]?.data || null,
    isLoading: (inferenceId: string) => inferenceCache[inferenceId]?.loading || false,
    getError: (inferenceId: string) => inferenceCache[inferenceId]?.error || null,
    openSheetInferenceId,
  };
}
