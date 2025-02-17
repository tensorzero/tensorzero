import { data } from "react-router";
import { useLoaderData } from "react-router";
import { DatasetBuilderForm } from "./DatasetBuilderForm";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";
import { getDatasetCounts } from "~/utils/clickhouse/datasets";

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

// TODO: Implement action to handle dataset creation/updates
export async function action() {
  // TODO: Handle form submission
  return data({ success: true });
}

export default function DatasetBuilder() {
  const { dataset_counts } = useLoaderData() as {
    dataset_counts: DatasetCountInfo[];
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <main>
        <h2 className="mb-4 text-2xl font-semibold">Dataset Builder</h2>
        <div className="mb-6 h-px w-full bg-gray-200"></div>
        <DatasetBuilderForm dataset_counts={dataset_counts} />
      </main>
    </div>
  );
}
