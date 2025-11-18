import { useState, useEffect, useRef, useCallback } from "react";
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

export function useInferenceHover(episodeRoute: string): {
  handleInferenceHover: (inferenceId: string) => void;
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
  const timeoutsRef = useRef<Record<string, ReturnType<typeof setTimeout>>>({});
  const activeFetchesRef = useRef<Set<string>>(new Set());

  const submitFetch = useCallback((inferenceId: string) => {
    // Cancel any active fetches if fetcher is busy
    if (fetcher.state !== "idle" && activeFetchesRef.current.size > 0) {
      const cancelledIds = Array.from(activeFetchesRef.current);
      activeFetchesRef.current.clear();
      
      setInferenceCache(prev => {
        const updated = { ...prev };
        cancelledIds.forEach(id => {
          if (updated[id]?.loading) {
            updated[id] = { ...updated[id], loading: false };
          }
        });
        return updated;
      });
    }
    
    setInferenceCache(prev => ({
      ...prev,
      [inferenceId]: { data: null, loading: true, error: null }
    }));
    
    activeFetchesRef.current.add(inferenceId);
    
    const formData = new FormData();
    formData.append("_action", "fetchInference");
    formData.append("inferenceId", inferenceId);
    
    fetcher.submit(formData, { 
      method: "POST",
      action: episodeRoute
    });
  }, [fetcher, episodeRoute]);

  const handleInferenceHover = useCallback((inferenceId: string) => {
    const timeout = timeoutsRef.current[inferenceId];
    if (timeout) clearTimeout(timeout);

    const currentState = inferenceCache[inferenceId];
    if (currentState?.data || currentState?.loading) return;

    timeoutsRef.current[inferenceId] = setTimeout(() => {
      submitFetch(inferenceId);
      delete timeoutsRef.current[inferenceId];
    }, 100);
  }, [inferenceCache, submitFetch]);

  const handleOpenSheet = useCallback((inferenceId: string) => {
    setOpenSheetInferenceId(inferenceId);
    
    const currentState = inferenceCache[inferenceId];
    if (!currentState?.data && !currentState?.loading) {
      submitFetch(inferenceId);
    }
  }, [inferenceCache, submitFetch]);

  const handleCloseSheet = useCallback(() => {
    setOpenSheetInferenceId(null);
  }, []);

  useEffect(() => {
    if (fetcher.state !== "idle" || !fetcher.data) return;
    
    const { inferenceId, inference, error } = fetcher.data;
    
    if (inferenceId && activeFetchesRef.current.has(inferenceId)) {
      activeFetchesRef.current.delete(inferenceId);
      
      setInferenceCache(prev => ({
        ...prev,
        [inferenceId]: {
          data: inference || null,
          loading: false,
          error: error || null
        }
      }));
    } else if (!inferenceId && activeFetchesRef.current.size > 0) {
      // Clear all loading states for active fetches on unknown error
      const activeIds = Array.from(activeFetchesRef.current);
      activeFetchesRef.current.clear();
      
      setInferenceCache(prev => {
        const updated = { ...prev };
        activeIds.forEach(id => {
          updated[id] = { ...updated[id], loading: false, error: 'Unknown error' };
        });
        return updated;
      });
    }
  }, [fetcher.state, fetcher.data]);

  useEffect(() => {
    return () => {
      // Clean up all timeouts on unmount
      Object.values(timeoutsRef.current).forEach(timeout => {
        clearTimeout(timeout);
      });
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