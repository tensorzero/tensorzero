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

export function useInferenceHover(episodeRoute: string): {
  handleInferenceHover: (inferenceId: string) => void;
  handleOpenSheet: (inferenceId: string) => void;
  handleCloseSheet: () => void;
  getInferenceData: (inferenceId: string) => ParsedInferenceRow | null;
  isLoading: (inferenceId: string) => boolean;
  getError: (inferenceId: string) => string | null;
  openSheetInferenceId: string | null;
} {
  const [fetchedInferenceId, setFetchedInferenceId] = useState<string | null>(null);
  const [openSheetInferenceId, setOpenSheetInferenceId] = useState<string | null>(null);
  const [inferenceCache, setInferenceCache] = useState<Record<string, InferenceState>>({});
  const fetcher = useFetcher<ActionData>();
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleInferenceHover = (inferenceId: string) => {
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
      
      setFetchedInferenceId(inferenceId);
      
      const formData = new FormData();
      formData.append("_action", "fetchInference");
      formData.append("inferenceId", inferenceId);
      
      fetcher.submit(formData, { 
        method: "POST",
        action: episodeRoute
      });
    }, 100);
  };

  const handleOpenSheet = (inferenceId: string) => {
    setOpenSheetInferenceId(inferenceId);
    
    // Fetch immediately if not in cache
    const currentState = inferenceCache[inferenceId];
    if (!currentState?.data && !currentState?.loading) {
      setInferenceCache(prev => ({
        ...prev,
        [inferenceId]: { data: null, loading: true, error: null }
      }));
      
      setFetchedInferenceId(inferenceId);
      
      const formData = new FormData();
      formData.append("_action", "fetchInference");
      formData.append("inferenceId", inferenceId);
      
      fetcher.submit(formData, { 
        method: "POST",
        action: episodeRoute
      });
    }
  };

  const handleCloseSheet = () => {
    setOpenSheetInferenceId(null);
  };

  useEffect(() => {
    if (fetcher.state === "idle" && fetcher.data && fetchedInferenceId) {
      setInferenceCache(prev => ({
        ...prev,
        [fetchedInferenceId]: {
          data: fetcher.data?.inference || null,
          loading: false,
          error: fetcher.data?.error || null
        }
      }));
    }
  }, [fetcher.state, fetcher.data, fetchedInferenceId]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return {
    handleInferenceHover,
    handleOpenSheet,
    handleCloseSheet,
    getInferenceData: (inferenceId: string) => inferenceCache[inferenceId]?.data || null,
    isLoading: (inferenceId: string) => inferenceCache[inferenceId]?.loading || false,
    getError: (inferenceId: string) => inferenceCache[inferenceId]?.error || null,
    openSheetInferenceId,
  };
}