import { useEffect } from "react";
import { useFetcher } from "react-router";
import type { ResolveUuidResponse } from "~/types/tensorzero";
import { toResolveUuidApi } from "~/utils/urls";

// Module-level cache survives component unmount/remount cycles.
// React Router deletes keyed fetcher data when all consumers unmount,
// so components that remount find the data gone. This cache ensures
// previously-resolved UUIDs are available immediately on remount.
const resolvedUuids = new Map<string, ResolveUuidResponse>();

/**
 * Resolves a UUID to its entity type. Uses a module-level cache to
 * provide instant results for previously-resolved UUIDs even when
 * components remount. The keyed fetcher handles the actual HTTP
 * request and deduplication.
 */
export function useResolveUuid(uuid: string): {
  data: ResolveUuidResponse | null;
  isLoading: boolean;
} {
  const normalizedUuid = uuid.toLowerCase();
  const fetcher = useFetcher<ResolveUuidResponse>({
    key: `resolve-uuid-${normalizedUuid}`,
  });

  // Populate cache when fetcher resolves with a successful result.
  // Empty object_types means the UUID was unknown or the request failed,
  // so we skip caching to allow retries on future mounts.
  if (
    fetcher.data &&
    fetcher.data.object_types.length > 0 &&
    !resolvedUuids.has(normalizedUuid)
  ) {
    resolvedUuids.set(normalizedUuid, fetcher.data);
  }

  useEffect(() => {
    if (
      fetcher.state === "idle" &&
      !fetcher.data &&
      !resolvedUuids.has(normalizedUuid)
    ) {
      fetcher.load(toResolveUuidApi(normalizedUuid));
    }
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [normalizedUuid]);

  const data = fetcher.data ?? resolvedUuids.get(normalizedUuid) ?? null;

  return {
    data,
    isLoading: !data && fetcher.state !== "idle",
  };
}
