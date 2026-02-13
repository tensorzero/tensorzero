import type { Route } from "./+types/episode_preview.route";
import { data } from "react-router";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";

export async function loader({ params }: Route.LoaderArgs) {
  const { episode_id } = params;
  if (!episode_id) {
    throw data("Episode ID is required", { status: 400 });
  }

  try {
    const client = getTensorZeroClient();
    const response = await client.getEpisodeInferenceCount(episode_id);
    return Response.json({
      inference_count: Number(response.inference_count),
    });
  } catch (error) {
    if (error instanceof Response) {
      throw error;
    }
    logger.error("Failed to fetch episode preview:", error);
    throw data("Failed to fetch episode preview", { status: 500 });
  }
}
