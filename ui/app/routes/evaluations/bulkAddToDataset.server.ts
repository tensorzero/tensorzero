import { data } from "react-router";
import { getConfig } from "~/utils/config/index.server";
import { handleAddToDatasetAction } from "~/utils/dataset.server";
import { logger } from "~/utils/logger";

export interface SelectedItemData {
  inference_id: string;
  variant_name: string;
  episode_id?: string;
}

export async function handleBulkAddToDataset(
  dataset: string,
  selectedItems: SelectedItemData[],
  evaluation_name: string,
) {
  const config = await getConfig();
  const evaluation_config = config.evaluations[evaluation_name];

  if (!evaluation_config) {
    return data(
      {
        error: `Evaluation config not found for ${evaluation_name}`,
        success: false,
      },
      { status: 404 },
    );
  }

  const function_name = evaluation_config.function_name;

  // Process all selected items in parallel
  const results = await Promise.allSettled(
    selectedItems.map((item) => {
      const itemFormData = new FormData();
      itemFormData.append("dataset", dataset);
      itemFormData.append("output", "inherit");
      itemFormData.append("inference_id", item.inference_id);
      itemFormData.append("function_name", function_name);
      itemFormData.append("variant_name", item.variant_name);
      itemFormData.append("episode_id", item.episode_id || "");
      itemFormData.append("_action", "addToDataset");

      return handleAddToDatasetAction(itemFormData).catch((error) => {
        logger.error(
          `Failed to add inference ${item.inference_id} to dataset:`,
          error,
        );
        throw error;
      });
    }),
  );

  const errors: string[] = [];
  let successCount = 0;

  results.forEach((result, index) => {
    if (result.status === "fulfilled") {
      successCount++;
    } else {
      errors.push(
        `Failed to add inference ${selectedItems[index].inference_id}`,
      );
    }
  });

  if (errors.length > 0 && successCount === 0) {
    return data(
      {
        error: `Failed to add all inferences: ${errors.join(", ")}`,
        success: false,
      },
      { status: 400 },
    );
  }

  return data({
    success: true,
    count: successCount,
    dataset: dataset,
    errors: errors.length > 0 ? errors : undefined,
  });
}
