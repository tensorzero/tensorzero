import { useEffect } from "react";
import { data, useFetcher, type LoaderFunctionArgs } from "react-router";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";
import { getDatasetCounts } from "~/utils/clickhouse/datasets.server";

export async function loader({ request }: LoaderFunctionArgs) {
  try {
    const url = new URL(request.url);
    const functionName = url.searchParams.get("function") ?? undefined;
    const datasetCounts = await getDatasetCounts({
      function_name: functionName,
    });
    const datasets = datasetCounts.map((d) => ({
      name: d.dataset_name,
      count: d.count,
      lastUpdated: d.last_updated,
    }));
    return data({ datasets });
  } catch (error) {
    return data({ error: `Failed to get count: ${error}` }, 500);
  }
}

/**
 * A hook that fetches the count of rows that would be inserted into a dataset based on the provided form values.
 * This hook automatically refetches when the form values change.
 *
 * @param control - The control object from react-hook-form
 * @returns An object containing:
 *  - count: Number of rows that would be inserted
 *  - isLoading: Whether the count is currently being fetched
 */
export function useDatasetCountFetcher(functionName: string | undefined): {
  datasets: DatasetCountInfo[] | null;
  isLoading: boolean;
} {
  const countFetcher = useFetcher();

  useEffect(() => {
    if (functionName) {
      const searchParams = new URLSearchParams();
      searchParams.set("function", functionName);
      countFetcher.load(`/api/datasets/count_inserts?${searchParams}`);
    }
    // TODO: Fix and stop ignoring lint rule
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [functionName]);

  return {
    datasets: countFetcher.data?.datasets ?? null,
    isLoading: countFetcher.state === "loading",
  };
}
