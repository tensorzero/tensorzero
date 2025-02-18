import { data } from "react-router";
import { useLoaderData } from "react-router";
import { DatasetBuilderForm } from "./DatasetBuilderForm";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";
import { getDatasetCounts } from "~/utils/clickhouse/datasets";
import type { CountsData } from "~/routes/api/curated_inferences/count.route";
import { useFetcher } from "react-router";
import { useEffect } from "react";

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

/**
 * A hook that fetches counts for inferences, feedbacks, and curated inferences based on function, metric, and threshold parameters.
 * This hook automatically refetches when any of the parameters change.
 *
 * @param params.functionName - The name of the function to get counts for
 * @param params.metricName - Optional metric name to filter counts by
 * @param params.threshold - Optional threshold value for curated inferences
 * @returns An object containing:
 *  - inferenceCount: Total number of inferences for the function
 *  - feedbackCount: Number of feedbacks for the function/metric combination
 *  - curatedInferenceCount: Number of curated inferences meeting the threshold criteria
 *  - isLoading: Whether the counts are currently being fetched
 */
export function useCountFetcher(params: {
  functionName?: string;
  metricName?: string;
  threshold?: number;
}): CountsData & { isLoading: boolean } {
  const countFetcher = useFetcher();

  useEffect(() => {
    if (params.functionName) {
      const searchParams = new URLSearchParams();
      searchParams.set("function", params.functionName);
      if (params.metricName) searchParams.set("metric", params.metricName);
      if (params.threshold)
        searchParams.set("threshold", String(params.threshold));

      countFetcher.load(`/api/curated_inferences/count?${searchParams}`);
    }
  }, [params.functionName, params.metricName, params.threshold]);

  return {
    inferenceCount: countFetcher.data?.inferenceCount ?? null,
    feedbackCount: countFetcher.data?.feedbackCount ?? null,
    curatedInferenceCount: countFetcher.data?.curatedInferenceCount ?? null,
    isLoading: countFetcher.state === "loading",
  };
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
