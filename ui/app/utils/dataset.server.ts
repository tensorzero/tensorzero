import { data } from "react-router";
import type { InferenceOutputSource } from "~/types/tensorzero";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";
import { toDatapointUrl } from "~/utils/urls";

type ActionData =
  | { redirectTo: string; error?: never }
  | { error: string; redirectTo?: never };

function toOutputSource(output: string): InferenceOutputSource {
  if (output === "inherit" || output === "inference") return "inference";
  if (output === "demonstration" || output === "none") return output;
  throw new Error(`Unknown output type ${output}`);
}

export async function handleAddToDatasetAction(formData: FormData) {
  const dataset = formData.get("dataset");
  const output = formData.get("output");
  const inferenceId = formData.get("inference_id");

  if (!dataset || !output || !inferenceId) {
    return data<ActionData>(
      { error: "Missing required fields" },
      { status: 400 },
    );
  }

  try {
    const result = await getTensorZeroClient().createDatapointsFromInferences(
      dataset.toString(),
      {
        type: "inference_ids",
        inference_ids: [inferenceId.toString()],
        output_source: toOutputSource(output.toString()),
      },
    );
    if (result.ids.length !== 1) {
      return data<ActionData>(
        { error: "Expected exactly one datapoint to be created" },
        { status: 500 },
      );
    }
    return data<ActionData>({
      redirectTo: toDatapointUrl(dataset.toString(), result.ids[0]),
    });
  } catch (error) {
    logger.error(error);
    return data<ActionData>(
      { error: `Failed to create datapoint: ${error}` },
      { status: 400 },
    );
  }
}
