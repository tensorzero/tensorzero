import { useEffect, useRef, useMemo, useState } from "react";
import { useFetcher, type ActionFunctionArgs } from "react-router";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type {
  CountInferencesRequest,
  InferenceFilter,
  InferenceOutputSource,
} from "~/types/tensorzero";

/// Count the number of inferences matching the given query parameters.
/// This is used by the dataset builder to show how many datapoints will be created.
export async function action({
  request,
}: ActionFunctionArgs): Promise<Response> {
  const body = await request.json();
  const countRequest: CountInferencesRequest = {
    function_name: body.function_name,
    variant_name: body.variant_name,
    episode_id: body.episode_id,
    output_source: body.output_source || "inference",
    filters: body.filters,
    search_query_experimental: body.search_query_experimental,
  };

  // Require function_name for now to avoid expensive full table scans
  if (!countRequest.function_name) {
    return Response.json({ count: 0, error: "Function name is required" });
  }

  try {
    const client = getTensorZeroClient();
    const count = await client.countInferences(countRequest);
    return Response.json({ count } as CountInferencesData);
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to count inferences";
    return Response.json({ count: 0, error: message } as CountInferencesData);
  }
}

export interface CountInferencesData {
  count: number;
  error?: string;
}

/**
 * A hook that fetches the count of inferences matching the given parameters.
 * Automatically debounces requests to avoid excessive API calls.
 *
 * @param params - The count inference parameters
 * @returns An object containing:
 *  - count: The count of matching inferences (null while loading)
 *  - isLoading: Whether the count is currently being fetched
 */
export function useInferenceCountFetcher(params: {
  functionName?: string;
  variantName?: string;
  episodeId?: string;
  outputSource?: InferenceOutputSource;
  filters?: InferenceFilter;
  searchQuery?: string;
  enabled?: boolean;
}): CountInferencesData & { isLoading: boolean } {
  const countFetcher = useFetcher<CountInferencesData>();
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastSubmittedRef = useRef<string | null>(null);
  const [isPendingDebounce, setIsPendingDebounce] = useState(false);

  // Memoize the request body to detect actual changes
  const requestBody = useMemo((): CountInferencesRequest | null => {
    // Don't create request body if disabled or no function name
    if (params.enabled === false || !params.functionName) return null;
    return {
      function_name: params.functionName,
      variant_name: params.variantName,
      episode_id: params.episodeId,
      output_source: params.outputSource || "inference",
      filters: params.filters,
      search_query_experimental: params.searchQuery,
    };
  }, [
    params.enabled,
    params.functionName,
    params.variantName,
    params.episodeId,
    params.outputSource,
    params.filters,
    params.searchQuery,
  ]);

  // Debounced fetch effect
  useEffect(() => {
    if (!requestBody) return;

    const requestBodyJson = JSON.stringify(requestBody);

    // Skip if same as last submitted
    if (requestBodyJson === lastSubmittedRef.current) return;

    // Mark as pending while debouncing
    setIsPendingDebounce(true);

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(() => {
      setIsPendingDebounce(false);
      lastSubmittedRef.current = requestBodyJson;
      countFetcher.submit(requestBodyJson, {
        method: "POST",
        action: "/api/inferences/count",
        encType: "application/json",
      });
    }, 300);

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [requestBody, countFetcher]);

  const isFetcherLoading =
    countFetcher.state === "loading" || countFetcher.state === "submitting";

  return {
    count: countFetcher.data?.count ?? 0,
    error: countFetcher.data?.error,
    isLoading: isPendingDebounce || isFetcherLoading,
  };
}
