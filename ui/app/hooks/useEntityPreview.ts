import { useEffect } from "react";
import { useFetcher } from "react-router";
import { toEpisodePreviewApi, toInferencePreviewApi } from "~/utils/urls";

export enum EntityPreviewType {
  Inference = "inference",
  Episode = "episode",
}

function getPreviewUrl(type: EntityPreviewType, id: string): string {
  switch (type) {
    case EntityPreviewType.Inference:
      return toInferencePreviewApi(id);
    case EntityPreviewType.Episode:
      return toEpisodePreviewApi(id);
  }
}

export function useEntityPreview<T>({
  type,
  id,
  enabled,
}: {
  type: EntityPreviewType;
  id: string;
  enabled: boolean;
}): { data: T | null; isLoading: boolean } {
  const url = getPreviewUrl(type, id);
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
