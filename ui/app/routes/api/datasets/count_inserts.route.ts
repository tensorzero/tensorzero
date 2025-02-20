import type { LoaderFunctionArgs } from "react-router";
import { countRowsForDataset } from "~/utils/clickhouse/datasets.server";
import { data } from "react-router";
import { serializedFormDataToDatasetQueryParams } from "~/routes/datasets/builder/types";
import { useEffect } from "react";
import { useFetcher } from "react-router";
import type { DatasetBuilderFormValues } from "~/routes/datasets/builder/types";
import { useWatch } from "react-hook-form";
import type { Control } from "react-hook-form";

/// Count the number of rows that would be inserted into the dataset
/// if a particular DatasetBuilderFormValues was submitted.
/// Since this is a GET request, the form data is passed in the URL under formData.
export async function loader({ request }: LoaderFunctionArgs) {
  const url = new URL(request.url);
  const formData = url.searchParams.get("formData");
  if (!formData) {
    return data({ error: "formData is required" }, { status: 400 });
  }

  try {
    const queryParams = serializedFormDataToDatasetQueryParams(formData);
    const count = await countRowsForDataset(queryParams);
    return data({ count });
  } catch (error) {
    return data({ error: `Invalid form data: ${error}` }, { status: 400 });
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
export function useDatasetCountFetcher(
  control: Control<DatasetBuilderFormValues>,
): {
  count: number | null;
  isLoading: boolean;
} {
  const countFetcher = useFetcher();
  const formValues = useWatch({ control });

  useEffect(() => {
    if (formValues && formValues.function !== undefined) {
      const searchParams = new URLSearchParams();
      searchParams.set("formData", JSON.stringify(formValues));
      countFetcher.load(`/api/datasets/count_inserts?${searchParams}`);
    }
  }, [formValues]);

  return {
    count: countFetcher.data?.count ?? null,
    isLoading:
      formValues.function !== undefined && countFetcher.state === "loading",
  };
}
