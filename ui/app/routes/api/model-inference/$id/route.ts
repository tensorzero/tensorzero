import { data, type LoaderFunctionArgs } from "react-router";
import { resolveModelInferences } from "~/utils/resolve.server";
import { logger } from "~/utils/logger";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { ParsedModelInferenceRow } from "~/utils/clickhouse/inference";

export interface ModelInferenceDetailData {
  model_inference: ParsedModelInferenceRow;
  inference_id: string;
}

export async function loader({
  params,
}: LoaderFunctionArgs): Promise<Response> {
  const { id } = params;

  if (!id) {
    throw data("Model inference ID is required", { status: 400 });
  }

  try {
    const client = getTensorZeroClient();

    const resolved = await client.resolveUuid(id);
    const modelInferenceObj = resolved.object_types.find(
      (obj: { type: string }) => obj.type === "model_inference",
    );

    if (!modelInferenceObj || modelInferenceObj.type !== "model_inference") {
      throw data(`UUID ${id} is not a model inference`, { status: 404 });
    }

    const { inference_id } = modelInferenceObj;

    const response = await client.getModelInferences(inference_id);
    const resolvedModelInferences = await resolveModelInferences(
      response.model_inferences,
    );

    const modelInference = resolvedModelInferences.find((mi) => mi.id === id);

    if (!modelInference) {
      throw data(`Model inference ${id} not found`, { status: 404 });
    }

    const result: ModelInferenceDetailData = {
      model_inference: modelInference,
      inference_id,
    };

    return Response.json(result);
  } catch (error) {
    if (error instanceof Response) {
      throw error;
    }
    logger.error("Failed to fetch model inference:", error);
    throw data("Failed to fetch model inference details", { status: 500 });
  }
}
