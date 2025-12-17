import { data, redirect } from "react-router";
import { DatasetBuilderForm } from "./DatasetBuilderForm";
import type { ActionFunctionArgs, RouteHandle } from "react-router";
import { formDataToFilterInferencesForDatasetBuilderRequest } from "./types";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { logger } from "~/utils/logger";
import { toDatasetUrl } from "~/utils/urls";
import { getTensorZeroClient } from "~/utils/get-tensorzero-client.server";

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
    const { datasetName, params } =
      formDataToFilterInferencesForDatasetBuilderRequest(jsonData);

    // Insert all matching inferences as datapoints (no deduplication)
    const client = getTensorZeroClient();
    const response = await client.insertFromMatchingInferences(
      datasetName,
      params,
    );

    return redirect(
      `${toDatasetUrl(datasetName)}?rowsAdded=${response.rows_inserted}`,
    );
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
