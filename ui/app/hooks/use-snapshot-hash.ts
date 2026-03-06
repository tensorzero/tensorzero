import { useSearchParams } from "react-router";

/**
 * Reads `?snapshot_hash` from the current URL.
 *
 * Config-only pages (functions, variants) receive the hash via URL from
 * whatever page linked to them. Components use this hook to propagate
 * the hash to outbound links and to show the SnapshotBanner.
 *
 * DB entities (inferences, episodes) carry their own snapshot_hash on
 * the object — those pages read it directly instead of using this hook.
 */
export function useSnapshotHash(): string | undefined {
  const [searchParams] = useSearchParams();
  return searchParams.get("snapshot_hash") ?? undefined;
}
