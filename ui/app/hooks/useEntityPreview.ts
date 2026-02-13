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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, enabled]);

  return {
    data: (fetcher.data as T) ?? null,
    isLoading:
      fetcher.state !== "idle" || (enabled && fetcher.data === undefined),
  };
}
