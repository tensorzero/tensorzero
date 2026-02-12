import { useEffect, useState } from "react";
import { useFetcher } from "react-router";
import type { ResolveUuidResponse } from "~/types/tensorzero";

/**
 * Module-level cache for UUID resolution results.
 * Persists across component instances within a session to avoid
 * redundant API calls for the same UUID.
 */
const resolveCache = new Map<string, ResolveUuidResponse>();

/**
 * Hook that resolves a UUID to determine what type of object it represents.
 * Uses a module-level cache to avoid duplicate fetches for the same UUID.
 *
 * Returns the cached result immediately if available, otherwise triggers
 * a fetch to the resolve_uuid API route.
 */
export function useResolveUuid(uuid: string): {
  data: ResolveUuidResponse | null;
  isLoading: boolean;
} {
  const cached = resolveCache.get(uuid);
  const [data, setData] = useState<ResolveUuidResponse | null>(cached ?? null);
  const fetcher = useFetcher<ResolveUuidResponse>();

  useEffect(() => {
    if (resolveCache.has(uuid)) {
      setData(resolveCache.get(uuid)!);
      return;
    }

    if (fetcher.state === "idle" && !fetcher.data) {
      fetcher.load(`/api/tensorzero/resolve_uuid/${encodeURIComponent(uuid)}`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uuid]);

  useEffect(() => {
    if (fetcher.data) {
      resolveCache.set(uuid, fetcher.data);
      setData(fetcher.data);
    }
  }, [fetcher.data, uuid]);

  return {
    data,
    isLoading: !data && fetcher.state !== "idle",
  };
}
