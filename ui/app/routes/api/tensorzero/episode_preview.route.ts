import type { Route } from "./+types/episode_preview.route";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";

// Returns preview data or null (never throws, to avoid crashing the
// page via the ErrorBoundary when loaded by a fetcher).
export async function loader({ params }: Route.LoaderArgs) {
  const { episode_id } = params;
  if (!episode_id) {
    return Response.json(null);
  }

  try {
    const client = getTensorZeroClient();
    const response = await client.getEpisodeInferenceCount(episode_id);
    return Response.json({
      inference_count: Number(response.inference_count),
    });
  } catch (error) {
    logger.error("Failed to fetch episode preview:", error);
    return Response.json(null);
  }
}
