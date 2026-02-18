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

interface EntityPreviewParams {
  type: EntityPreviewType;
  id: string;
  enabled: boolean;
}

interface EntityPreviewResult<T> {
  data: T | null;
  isLoading: boolean;
}

export function useEntityPreview<T>({
  type,
  id,
  enabled,
}: EntityPreviewParams): EntityPreviewResult<T> {
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
    // Treat "not yet fetched" as loading when enabled, so callers
    // show a skeleton instead of nothing on the first render before
    // the effect fires.
    isLoading:
      fetcher.state !== "idle" || (enabled && fetcher.data === undefined),
  };
}
