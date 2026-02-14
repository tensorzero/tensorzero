import { useEffect } from "react";
import { useFetcher } from "react-router";
import type { ResolveUuidResponse } from "~/types/tensorzero";

/**
 * Resolves a UUID to its entity type. Keyed fetcher deduplicates
 * concurrent requests and caches results across component instances.
 */
export function useResolveUuid(uuid: string): {
  data: ResolveUuidResponse | null;
  isLoading: boolean;
} {
  const normalizedUuid = uuid.toLowerCase();
  const fetcher = useFetcher<ResolveUuidResponse>({
    key: `resolve-uuid-${normalizedUuid}`,
  });

  useEffect(() => {
    if (fetcher.state === "idle" && !fetcher.data) {
      fetcher.load(
        `/api/tensorzero/resolve_uuid/${encodeURIComponent(normalizedUuid)}`,
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [normalizedUuid]);

  return {
    data: fetcher.data ?? null,
    isLoading: fetcher.state !== "idle",
  };
}
