import { useState, useEffect, useRef } from "react";
import { useFetcher } from "react-router";
import type { ParsedInferenceRow } from "~/utils/clickhouse/inference";

interface ActionData {
  inference?: ParsedInferenceRow;
  error?: string;
}

export function useInferenceHover(episodeRoute: string) {
  const [hoveredInferenceId, setHoveredInferenceId] = useState<string | null>(null);
  const [inferenceData, setInferenceData] = useState<Record<string, ParsedInferenceRow>>({});
  const [loadingInferences, setLoadingInferences] = useState<Set<string>>(new Set());
  const fetcher = useFetcher<ActionData>();
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleInferenceHover = (inferenceId: string) => {
    setHoveredInferenceId(inferenceId);
    
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    if (inferenceData[inferenceId] || loadingInferences.has(inferenceId)) {
      return;
    }

    // Debounce the fetch request by 300ms
    timeoutRef.current = setTimeout(() => {
      setLoadingInferences(prev => new Set([...prev, inferenceId]));
      
      const formData = new FormData();
      formData.append("_action", "fetchInference");
      formData.append("inferenceId", inferenceId);
      
      fetcher.submit(formData, { 
        method: "POST",
        action: episodeRoute
      });
    }, 300);
  };

  useEffect(() => {
    if (fetcher.state === "idle" && fetcher.data && hoveredInferenceId) {
       setLoadingInferences(prev => {
          const newSet = new Set(prev);
          newSet.delete(hoveredInferenceId);
          return newSet;
        });

      if (fetcher.data.inference) {
        setInferenceData(prev => ({
          ...prev,
          [hoveredInferenceId]: fetcher.data!.inference!
        }));
      }
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
    getInferenceData: (inferenceId: string) => inferenceData[inferenceId] || null,
    isLoading: (inferenceId: string) => loadingInferences.has(inferenceId),
  };
}