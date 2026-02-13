import type { Route } from "./+types/inference_preview.route";
import { data } from "react-router";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";

export async function loader({ params }: Route.LoaderArgs) {
  const { inference_id } = params;
  if (!inference_id) {
    throw data("Inference ID is required", { status: 400 });
  }

  try {
    const client = getTensorZeroClient();
    const response = await client.getInferences({
      ids: [inference_id],
      output_source: "inference",
    });

    if (response.inferences.length === 0) {
      throw data("Inference not found", { status: 404 });
    }

    const inference = response.inferences[0];
    return Response.json({
      inference_id: inference.inference_id,
      function_name: inference.function_name,
      variant_name: inference.variant_name,
      episode_id: inference.episode_id,
      timestamp: inference.timestamp,
      processing_time_ms: inference.processing_time_ms
        ? Number(inference.processing_time_ms)
        : null,
      type: inference.type,
    });
  } catch (error) {
    if (error instanceof Response) {
      throw error;
    }
    logger.error("Failed to fetch inference preview:", error);
    throw data("Failed to fetch inference preview", { status: 500 });
  }
}
