import type { Route } from "./+types/inference_sheet.route";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";

// Returns StoredInference or null (never throws, to avoid crashing the
// autopilot page via the ErrorBoundary when loaded by a fetcher).
export async function loader({ params }: Route.LoaderArgs) {
  const { inference_id } = params;
  if (!inference_id) {
    return Response.json(null);
  }

  try {
    const client = getTensorZeroClient();
    const response = await client.getInferences({
      ids: [inference_id],
      output_source: "inference",
    });

    if (response.inferences.length === 0) {
      return Response.json(null);
    }

    return Response.json(response.inferences[0]);
  } catch (error) {
    logger.error("Failed to fetch inference for side sheet:", error);
    return Response.json(null);
  }
}
