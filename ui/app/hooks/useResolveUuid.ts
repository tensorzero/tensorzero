import { useEffect } from "react";
import { useFetcher } from "react-router";
import type { ResolveUuidResponse } from "~/types/tensorzero";

/**
 * Hook that resolves a UUID to determine what type of object it represents.
 *
 * Uses a keyed `useFetcher` so that all instances resolving the same UUID
 * share a single fetch and cached result via React Router's global state.
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
    // Intentionally omit `fetcher` — including it causes infinite re-fetch loops
    // because each fetch mutates fetcher state, which would re-trigger the effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [normalizedUuid]);

  return {
    data: fetcher.data ?? null,
    isLoading: fetcher.state !== "idle",
  };
}
