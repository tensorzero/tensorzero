import { useEffect } from "react";
import { useFetcher } from "react-router";

export function useEntityPreview<T>(
  url: string,
  enabled: boolean,
): { data: T | null; isLoading: boolean } {
  const fetcher = useFetcher<T>({ key: url });

  useEffect(() => {
    if (enabled && fetcher.state === "idle" && !fetcher.data) {
      fetcher.load(url);
    }
    // Intentionally omits fetcher from deps to avoid infinite loops.
    // Re-runs when url or enabled changes, which covers the hovercard
    // open/close lifecycle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, enabled]);

  return {
    data: (fetcher.data as T) ?? null,
    isLoading: fetcher.state !== "idle",
  };
}
