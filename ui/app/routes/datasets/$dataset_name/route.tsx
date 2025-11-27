import { getDatasetMetadata } from "~/utils/clickhouse/datasets.server";
import type { Route } from "./+types/route";
import DatasetRowTable from "./DatasetRowTable";
import { data, isRouteErrorResponse, redirect } from "react-router";
import { useNavigate } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import DatasetRowSearchBar from "./DatasetRowSearchBar";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { useToast } from "~/hooks/use-toast";
import { useEffect } from "react";
import { logger } from "~/utils/logger";
import { getNativeTensorZeroClient } from "~/utils/tensorzero/native_client.server";
import { useFetcher } from "react-router";
import { DeleteButton } from "~/components/utils/DeleteButton";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import { useReadOnly } from "~/context/read-only";

export async function loader({ request, params }: Route.LoaderArgs) {
  const { dataset_name } = params;
  const url = new URL(request.url);
  const limit = Number(url.searchParams.get("limit")) || 15;
  const offset = Number(url.searchParams.get("offset")) || 0;
  const rowsAddedParam = url.searchParams.get("rowsAdded");
  const rowsSkippedParam = url.searchParams.get("rowsSkipped");
  const rowsAdded = rowsAddedParam !== null ? Number(rowsAddedParam) : null;
  const rowsSkipped =
    rowsSkippedParam !== null ? Number(rowsSkippedParam) : null;

  if (limit > 100) {
    throw data("Limit cannot exceed 100", { status: 400 });
  }

  const [counts, getDatapointsResponse] = await Promise.all([
    getDatasetMetadata({}),
    getTensorZeroClient().listDatapoints(dataset_name, { limit, offset }),
  ]);
  const count_info = counts.find(
    (count) => count.dataset_name === dataset_name,
  );
  if (!count_info) {
    throw data("Dataset not found", { status: 404 });
  }
  const rows = getDatapointsResponse.datapoints;
  return { rows, count_info, limit, offset, rowsAdded, rowsSkipped };
}

export async function action({ request, params }: Route.ActionArgs) {
  const { dataset_name } = params;
  const formData = await request.formData();
  const action = formData.get("action");

  if (action === "delete") {
    if (!dataset_name) {
      throw data("Dataset name is required", { status: 400 });
    }
    const client = await getNativeTensorZeroClient();
    await client.staleDataset(dataset_name);
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
    const counts = await getDatasetMetadata({});
    const count_info = counts.find(
      (count) => count.dataset_name === dataset_name,
    );

    // If no datapoints remain, redirect to datasets list
    if (!count_info || count_info.count === 0) {
      return redirect("/datasets");
    }

    return { success: true };
  }

  return null;
}

export default function DatasetDetailPage({
  loaderData,
}: Route.ComponentProps) {
  const { rows, count_info, limit, offset, rowsAdded, rowsSkipped } =
    loaderData;
  const { toast } = useToast();
  const isReadOnly = useReadOnly();
  const fetcher = useFetcher();
  const navigate = useNavigate();

  // Use useEffect to show toast only after component mounts
  useEffect(() => {
    if (rowsAdded !== null) {
      const { dismiss } = toast.success({
        title: "Dataset Updated",
        description: `Added ${rowsAdded} rows to the dataset. Skipped ${rowsSkipped} duplicate rows.`,
      });
      return () => dismiss({ immediate: true });
    }
    return;
    // TODO: Fix and stop ignoring lint rule
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rowsAdded, toast]);

  const handleDelete = () => {
    const formData = new FormData();
    formData.append("action", "delete");
    fetcher.submit(formData, { method: "post" });
  };
  const handleNextPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset + limit));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };
  const handlePreviousPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset - limit));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <PageLayout>
      <PageHeader
        heading={`Dataset`}
        name={count_info.dataset_name}
        count={count_info.count}
      >
        <div className="flex justify-start">
          <DeleteButton
            onClick={handleDelete}
            isLoading={fetcher.state === "submitting"}
            disabled={isReadOnly}
          />
        </div>
      </PageHeader>

      <SectionLayout>
        <DatasetRowSearchBar dataset_name={count_info.dataset_name} />
        <DatasetRowTable rows={rows} dataset_name={count_info.dataset_name} />
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={offset === 0}
          disableNext={offset + limit >= count_info.count}
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
