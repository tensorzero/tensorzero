import { useEffect } from "react";
import { useFetcher } from "react-router";

export interface EpisodePreview {
  inference_count: number;
}

export function useEpisodePreview(
  episodeId: string,
  enabled: boolean,
): { data: EpisodePreview | null; isLoading: boolean } {
  const fetcher = useFetcher<EpisodePreview>({
    key: `episode-preview-${episodeId}`,
  });

  useEffect(() => {
    if (enabled && fetcher.state === "idle" && !fetcher.data) {
      fetcher.load(
        `/api/tensorzero/episode_preview/${encodeURIComponent(episodeId)}`,
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [episodeId, enabled]);

  return {
    data: fetcher.data ?? null,
    isLoading: fetcher.state !== "idle",
  };
}
