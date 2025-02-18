import { data } from "react-router";
import { useLoaderData } from "react-router";
import { DatasetBuilderForm } from "./DatasetBuilderForm";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";
import {
  getDatasetCounts,
  countRowsForDataset,
  insertRowsForDataset,
  DatasetQueryParamsSchema,
} from "~/utils/clickhouse/datasets";
import type { ActionFunctionArgs } from "react-router";
import { getComparisonOperator } from "~/utils/config/metric";
import { getInferenceJoinKey } from "~/utils/clickhouse/curation";

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
    const parsedData = JSON.parse(jsonData);
    console.log(parsedData);
    // Build and validate DatasetQueryParams from form data
    const queryParamsResult = DatasetQueryParamsSchema.safeParse({
      inferenceType: parsedData.type,
      function_name: parsedData.function,
      variant_name: parsedData.variant,
      dataset_name: parsedData.dataset,
      join_demonstrations: parsedData.join_demonstrations,
      extra_where: [],
      extra_params: {},
      ...(parsedData.metric_name && parsedData.threshold
        ? {
            metric_filter: {
              metric: parsedData.metric_name,
              metric_type: parsedData.metric_config?.type,
              operator: getComparisonOperator(
                parsedData.metric_config?.optimize,
              ),
              threshold: parsedData.threshold,
              join_on: getInferenceJoinKey(parsedData.metric_config?.level),
            },
          }
        : {}),
    });

    if (!queryParamsResult.success) {
      return data(
        { errors: { message: queryParamsResult.error.message } },
        { status: 400 },
      );
    }
    // TODO: make this one database call
    // Count rows and insert them concurrently
    const [count] = await Promise.all([
      countRowsForDataset(queryParamsResult.data),
      insertRowsForDataset(queryParamsResult.data),
    ]);

    return data({ success: true, count });
  } catch (error) {
    console.error("Error creating dataset:", error);
    return data(
      { errors: { message: "Error creating dataset" } },
      { status: 500 },
    );
  }
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
