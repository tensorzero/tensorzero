import type { Route } from "./+types/route";
import DatasetRowTable, { type DatasetRowsData } from "./DatasetRowTable";
import { data, isRouteErrorResponse, redirect } from "react-router";
import DatasetRowSearchBar from "./DatasetRowSearchBar";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { useToast } from "~/hooks/use-toast";
import { useEffect } from "react";
import { logger } from "~/utils/logger";
import { useFetcher } from "react-router";
import { DeleteButton } from "~/components/utils/DeleteButton";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import { useReadOnly } from "~/context/read-only";
import type { DatapointFilter } from "~/types/tensorzero";

export async function loader({ request, params }: Route.LoaderArgs) {
  const { dataset_name } = params;
  const url = new URL(request.url);
  const limit = Number(url.searchParams.get("limit")) || 15;
  const offset = Number(url.searchParams.get("offset")) || 0;
  const rowsAddedParam = url.searchParams.get("rowsAdded");
  const rowsAdded = rowsAddedParam !== null ? Number(rowsAddedParam) : null;
  const function_name = url.searchParams.get("function_name") || undefined;
  const search_query = url.searchParams.get("search_query") || undefined;
  const filterParam = url.searchParams.get("filter");
  let filter: DatapointFilter | undefined;
  if (filterParam) {
    try {
      filter = JSON.parse(filterParam) as DatapointFilter;
    } catch {
      // Invalid JSON, ignore
    }
  }

  if (limit > 100) {
    throw data("Limit cannot exceed 100", { status: 400 });
  }

  // Promise for count (streams to PageHeader)
  const countPromise = getTensorZeroClient()
    .listDatasets({})
    .then((response) => {
      const info = response.datasets.find(
        (d) => d.dataset_name === dataset_name,
      );
      if (!info) {
        throw Error("Dataset not found");
      }
      return info.datapoint_count;
    });

  // Promise for table data (streams to DatasetRowTable)
  const dataPromise: Promise<DatasetRowsData> = (async () => {
    const response = await getTensorZeroClient().listDatapoints(dataset_name, {
      limit: limit + 1, // Request one extra to check if there are more
      offset,
      function_name,
      search_query_experimental:
        search_query && search_query.length > 0 ? search_query : undefined,
      filter,
    });
    const allRows = response.datapoints;
    const hasMore = allRows.length > limit;
    const rows = hasMore ? allRows.slice(0, limit) : allRows;
    return { rows, hasMore };
  })();

  return {
    dataset_name,
    countPromise,
    dataPromise,
    limit,
    offset,
    rowsAdded,
    function_name,
    search_query,
    filter,
  };
}

export async function action({ request, params }: Route.ActionArgs) {
  const { dataset_name } = params;
  const formData = await request.formData();
  const action = formData.get("action");

  if (action === "delete") {
    if (!dataset_name) {
      throw data("Dataset name is required", { status: 400 });
    }
    const client = getTensorZeroClient();
    await client.deleteDataset(dataset_name);
    // Redirect to datasets list after successful deletion
    return redirect("/datasets");
  }

  if (action === "delete_datapoint") {
    const datapoint_id = formData.get("datapoint_id") as string;
    const function_name = formData.get("function_name");
    const function_type = formData.get("function_type");

    if (!datapoint_id || !function_name || !function_type) {
      throw data("Missing required fields for datapoint deletion", {
        status: 400,
      });
    }

    const config = await getConfig();
    const functionConfig = await getFunctionConfig(
      function_name as string,
      config,
    );
    if (!functionConfig) {
      throw data("Function configuration not found", { status: 404 });
    }

    await getTensorZeroClient().deleteDatapoints(dataset_name, [datapoint_id]);

    // Check if this was the last datapoint in the dataset
    const datasetMetadata = await getTensorZeroClient().listDatasets({});
    const count_info = datasetMetadata.datasets.find(
      (dataset) => dataset.dataset_name === dataset_name,
    );

    // If no datapoints remain, redirect to datasets list
    if (!count_info || count_info.datapoint_count === 0) {
      return redirect("/datasets");
    }

    return { success: true };
  }

  return null;
}

export default function DatasetDetailPage({
  loaderData,
}: Route.ComponentProps) {
  const {
    dataset_name,
    countPromise,
    dataPromise,
    limit,
    offset,
    rowsAdded,
    function_name,
    search_query,
    filter,
  } = loaderData;
  const { toast } = useToast();
  const isReadOnly = useReadOnly();
  const fetcher = useFetcher();

  // Use useEffect to show toast only after component mounts
  useEffect(() => {
    if (rowsAdded !== null) {
      const { dismiss } = toast.success({
        title: "Dataset Updated",
        description: `Added ${rowsAdded} rows to the dataset.`,
      });
      return () => dismiss({ immediate: true });
    }
    return;
    // TODO: Fix and stop ignoring lint rule
  }, [rowsAdded, toast]);

  const handleDelete = () => {
    const formData = new FormData();
    formData.append("action", "delete");
    fetcher.submit(formData, { method: "post" });
  };

  return (
    <PageLayout>
      <PageHeader heading={`Dataset`} name={dataset_name} count={countPromise}>
        <div className="flex justify-start">
          <DeleteButton
            onClick={handleDelete}
            isLoading={fetcher.state === "submitting"}
            disabled={isReadOnly}
          />
        </div>
      </PageHeader>

      <SectionLayout>
        <DatasetRowSearchBar dataset_name={dataset_name} />
        <DatasetRowTable
          data={dataPromise}
          dataset_name={dataset_name}
          limit={limit}
          offset={offset}
          function_name={function_name}
          search_query={search_query}
          filter={filter}
        />
      </SectionLayout>
    </PageLayout>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  logger.error(error);

  if (isRouteErrorResponse(error)) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">
          {error.status} {error.statusText}
        </h1>
        <p>{error.data}</p>
      </div>
    );
  } else if (error instanceof Error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">Error</h1>
        <p>{error.message}</p>
      </div>
    );
  } else {
    return (
      <div className="flex h-screen items-center justify-center text-red-500">
        <h1 className="text-2xl font-bold">Unknown Error</h1>
      </div>
    );
  }
}
