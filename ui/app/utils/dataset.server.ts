import { data } from "react-router";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";
import { toDatapointUrl } from "~/utils/urls";

type ActionData =
  | { redirectTo: string; error?: never }
  | { error: string; redirectTo?: never };

export async function handleAddToDatasetAction(formData: FormData) {
  const dataset = formData.get("dataset");
  const output = formData.get("output");
  const inferenceId = formData.get("inference_id");
  const functionName = formData.get("function_name");
  const variantName = formData.get("variant_name");
  const episodeId = formData.get("episode_id");

  if (
    !dataset ||
    !output ||
    !inferenceId ||
    !functionName ||
    !variantName ||
    !episodeId
  ) {
    return data<ActionData>(
      { error: "Missing required fields" },
      { status: 400 },
    );
  }

  try {
    const datapoint =
      await getTensorZeroClient().createDatapointFromInferenceLegacy(
        dataset.toString(),
        inferenceId.toString(),
        output.toString() as "inherit" | "demonstration" | "none",
        functionName.toString(),
        variantName.toString(),
        episodeId.toString(),
      );
    return data<ActionData>({
      redirectTo: toDatapointUrl(dataset.toString(), datapoint.id),
    });
  } catch (error) {
    logger.error(error);
    return data<ActionData>(
      {
        error:
          "Failed to create datapoint as a datapoint exists with the same `source_inference_id`",
      },
      { status: 400 },
    );
  }
}
