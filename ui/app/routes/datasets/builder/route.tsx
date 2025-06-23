import { data, redirect } from "react-router";
import { DatasetBuilderForm } from "./DatasetBuilderForm";
import {
  countRowsForDataset,
  getDatasetCounts,
  insertRowsForDataset,
} from "~/utils/clickhouse/datasets.server";
import type { ActionFunctionArgs, RouteHandle } from "react-router";
import { serializedFormDataToDatasetQueryParams } from "./types";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import type { Route } from "./+types/route";

export const handle: RouteHandle = {
  crumb: () => ["Builder"],
};

export async function loader() {
  const dataset_counts = await getDatasetCounts();
  return data({ dataset_counts });
}

export async function action({ request }: ActionFunctionArgs) {
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

    return redirect(
      `/datasets/${queryParams.dataset_name}?rowsAdded=${writtenRows}&rowsSkipped=${skippedRows}`,
    );
  } catch (error) {
    console.error("Error creating dataset:", error);
    return data({ errors: { message: `${error}` } }, { status: 500 });
  }
}

export default function DatasetBuilder({ loaderData }: Route.ComponentProps) {
  const { dataset_counts } = loaderData;

  return (
    <PageLayout>
      <PageHeader heading="Dataset Builder" />
      <SectionLayout>
        <DatasetBuilderForm dataset_counts={dataset_counts} />
      </SectionLayout>
    </PageLayout>
  );
}
