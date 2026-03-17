import { type ActionFunctionArgs } from "react-router";
import { handleAddToDatasetAction } from "~/utils/dataset.server";

/**
 * Creates a datapoint from an existing inference.
 *
 * Expected form data:
 * - dataset: string - The dataset name to add the datapoint to
 * - inference_id: string - The source inference ID
 * - output: "inherit" | "demonstration" | "none" - How to handle the output
 */
export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();
  return handleAddToDatasetAction(formData);
}
