import { data, redirect } from "react-router";
import { DatasetBuilderForm } from "./DatasetBuilderForm";
import {
  countRowsForDataset,
  insertRowsForDataset,
} from "~/utils/clickhouse/datasets.server";
import type { ActionFunctionArgs, RouteHandle } from "react-router";
import { serializedFormDataToDatasetQueryParams } from "./types";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { logger } from "~/utils/logger";
import { toDatasetUrl } from "~/utils/urls";
import { protectAction } from "~/utils/read-only.server";

export const handle: RouteHandle = {
  crumb: () => ["Builder"],
};

export async function action({ request }: ActionFunctionArgs) {
  protectAction();
  const formData = await request.formData();
  const jsonData = formData.get("data");

  if (!jsonData || typeof jsonData !== "string") {
    return data({ errors: { message: "Invalid form data" } }, { status: 400 });
  }

  try {
    const queryParams = serializedFormDataToDatasetQueryParams(jsonData);

    const [writtenRows, totalRows] = await Promise.all([
      insertRowsForDataset(queryParams),
      countRowsForDataset(queryParams),
    ]);
    const skippedRows = totalRows - writtenRows;

    if (!queryParams.dataset_name) {
      throw new Error("Dataset name is required");
    }

    return redirect(
      `${toDatasetUrl(queryParams.dataset_name)}?rowsAdded=${writtenRows}&rowsSkipped=${skippedRows}`,
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
