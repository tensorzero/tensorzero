import type { Route } from "./+types/inference_sheet.route";
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

    return Response.json(response.inferences[0]);
  } catch (error) {
    if (error instanceof Response) {
      throw error;
    }
    logger.error("Failed to fetch inference for side sheet:", error);
    throw data("Failed to fetch inference", { status: 500 });
  }
}
