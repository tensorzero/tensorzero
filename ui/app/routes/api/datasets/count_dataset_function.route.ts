import type { LoaderFunctionArgs } from "react-router";
import { countDatapointsForDatasetFunction } from "~/utils/clickhouse/datasets.server";
import { data } from "react-router";
import { useEffect } from "react";
import { useFetcher } from "react-router";

/// Count the number of rows that are available for a given dataset and function.
export async function loader({ params }: LoaderFunctionArgs) {
  const { dataset_name, function_name } = params;
  if (!dataset_name || !function_name) {
    return data(
      { error: "dataset_name and function_name are required" },
      { status: 400 },
    );
  }

  try {
    const count = await countDatapointsForDatasetFunction(
      dataset_name,
      function_name,
    );
    return data({ count });
  } catch (error) {
    return data({ error: `Failed to get count: ${error}` }, { status: 500 });
  }
}

/**
 * A hook that fetches the count of datapoints for a given dataset and function.
 * This hook automatically refetches when the dataset_name or function_name changes.
 *
 * @param dataset_name - The name of the dataset
 * @param function_name - The name of the function
 * @returns An object containing:
 *  - count: Number of datapoints for the given dataset and function
 *  - isLoading: Whether the count is currently being fetched
 */
export function useDatasetCountFetcher(
  dataset_name: string | null,
  function_name: string | null,
): {
  count: number | null;
  isLoading: boolean;
} {
  const countFetcher = useFetcher();

  useEffect(() => {
    if (dataset_name && function_name) {
      countFetcher.load(
        `/api/datasets/count/dataset/${dataset_name}/function/${function_name}`,
      );
    }
    // TODO: Fix and stop ignoring lint rule
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset_name, function_name]);

  return {
    count: countFetcher.data?.count ?? null,
    isLoading:
      dataset_name !== null &&
      function_name !== null &&
      countFetcher.state === "loading",
  };
}
