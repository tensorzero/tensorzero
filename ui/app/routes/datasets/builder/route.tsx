import { data, redirect } from "react-router";
import { DatasetBuilderForm } from "./DatasetBuilderForm";
import type { ActionFunctionArgs, RouteHandle } from "react-router";
import { DatasetBuilderFormValuesSchema } from "./types";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { logger } from "~/utils/logger";
import { toDatasetUrl } from "~/utils/urls";
import { getTensorZeroClient } from "~/utils/get-tensorzero-client.server";
import type { CreateDatapointsFromInferenceRequest } from "~/types/tensorzero";

export const handle: RouteHandle = {
  crumb: () => ["Builder"],
};

export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();
  const jsonData = formData.get("data");

  if (!jsonData || typeof jsonData !== "string") {
    return data({ errors: { message: "Invalid form data" } }, { status: 400 });
  }

  try {
    const parsedData = JSON.parse(jsonData);
    const formValues = DatasetBuilderFormValuesSchema.parse(parsedData);

    const datasetName = formValues.dataset;
    if (!datasetName) {
      throw new Error("Dataset name is required");
    }

    const functionName = formValues.function;
    if (!functionName) {
      throw new Error("Function name is required");
    }

    // Build the request for the from_inferences API
    const apiRequest: CreateDatapointsFromInferenceRequest = {
      type: "inference_query",
      function_name: functionName,
      variant_name: formValues.variant_name,
      episode_id: formValues.episode_id,
      search_query_experimental: formValues.search_query,
      filters: formValues.filters,
      output_source: formValues.output_source,
    };

    const client = getTensorZeroClient();
    const response = await client.createDatapointsFromInferences(
      datasetName,
      apiRequest,
    );

    const rowsAdded = response.ids.length;

    if (rowsAdded === 0) {
      return data(
        {
          errors: {
            message:
              "No matching inferences found. Try adjusting your filters.",
          },
        },
        { status: 400 },
      );
    }

    return redirect(`${toDatasetUrl(datasetName)}?rowsAdded=${rowsAdded}`);
  } catch (error) {
    logger.error("Error creating dataset:", error);
    return data({ errors: { message: `${error}` } }, { status: 500 });
  }
}

export default function DatasetBuilder() {
  return (
    <PageLayout>
      <PageHeader heading="Dataset Builder" />
      <SectionLayout>
        <DatasetBuilderForm />
      </SectionLayout>
    </PageLayout>
  );
}
