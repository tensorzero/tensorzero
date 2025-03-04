import { data, redirect } from "react-router";
import { useLoaderData } from "react-router";
import { DatasetBuilderForm } from "./DatasetBuilderForm";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";
import {
  getDatasetCounts,
  insertRowsForDataset,
} from "~/utils/clickhouse/datasets.server";
import type { ActionFunctionArgs } from "react-router";
import { serializedFormDataToDatasetQueryParams } from "./types";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";

export const meta = () => {
  return [
    { title: "TensorZero Dataset Builder" },
    {
      name: "description",
      content: "Dataset Builder",
    },
  ];
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

    await insertRowsForDataset(queryParams);

    return redirect(`/datasets/${queryParams.dataset_name}`);
  } catch (error) {
    console.error("Error creating dataset:", error);
    return data({ errors: { message: `${error}` } }, { status: 500 });
  }
}

export default function DatasetBuilder() {
  const { dataset_counts } = useLoaderData() as {
    dataset_counts: DatasetCountInfo[];
  };

  return (
    <div className="container mx-auto px-4 pb-8">
      <PageLayout>
        <PageHeader heading="Dataset Builder" />
        <SectionLayout>
          <DatasetBuilderForm dataset_counts={dataset_counts} />
        </SectionLayout>
      </PageLayout>
    </div>
  );
}
