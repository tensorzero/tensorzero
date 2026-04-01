import type { Route } from "./+types/route";
import DatasetRowTable, { type DatasetRowsData } from "./DatasetRowTable";
import { data, redirect } from "react-router";
import DatasetRowSearchBar from "./DatasetRowSearchBar";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
  Breadcrumbs,
} from "~/components/layout/PageLayout";
import { useToast } from "~/hooks/use-toast";
import { Suspense, useEffect } from "react";
import { Await, useFetcher } from "react-router";
import { ActionBar } from "~/components/layout/ActionBar";
import { AskAutopilotButton } from "~/components/autopilot/AskAutopilotButton";
import { DeleteButton } from "~/components/utils/DeleteButton";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import { useReadOnly } from "~/context/read-only";
import type { Datapoint, DatapointFilter } from "~/types/tensorzero";
import { TypeChat, TypeJson } from "~/components/icons/Icons";
import { StatsBar, StatsBarSkeleton } from "~/components/ui/StatsBar";

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

  // Check if dataset exists first (throws 404 if not found)
  const datasetsResponse = await getTensorZeroClient().listDatasets({});
  const datasetInfo = datasetsResponse.datasets.find(
    (d) => d.dataset_name === dataset_name,
  );
  if (!datasetInfo) {
    throw data(`Dataset "${dataset_name}" not found`, { status: 404 });
  }

  // Promise for count (streams to PageHeader)
  const countPromise = Promise.resolve(datasetInfo.datapoint_count);

  // Promise for stats (function/type breakdown from a sample)
  const statsPromise: Promise<DatasetStats> = (async () => {
    const response = await getTensorZeroClient().listDatapoints(dataset_name, {
      limit: 200,
      offset: 0,
    });
    return computeDatasetStats(
      response.datapoints,
      datasetInfo.datapoint_count,
      datasetInfo.last_updated,
    );
  })();

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
    statsPromise,
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

interface DatasetStats {
  totalCount: number;
  lastUpdated: string;
  functionBreakdown: { name: string; count: number }[];
  chatCount: number;
  jsonCount: number;
  withOutput: number;
  withoutOutput: number;
}

function computeDatasetStats(
  datapoints: Datapoint[],
  totalCount: number,
  lastUpdated: string,
): DatasetStats {
  const functionCounts = new Map<string, number>();
  let chatCount = 0;
  let jsonCount = 0;
  let withOutput = 0;
  let withoutOutput = 0;

  for (const dp of datapoints) {
    const fn = dp.function_name;
    functionCounts.set(fn, (functionCounts.get(fn) ?? 0) + 1);
    if (dp.type === "chat") chatCount++;
    else jsonCount++;
    if (dp.output != null) withOutput++;
    else withoutOutput++;
  }

  const functionBreakdown = Array.from(functionCounts.entries())
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count);

  return {
    totalCount,
    lastUpdated,
    functionBreakdown,
    chatCount,
    jsonCount,
    withOutput,
    withoutOutput,
  };
}

function DatasetStatsBar({ stats }: { stats: DatasetStats }) {
  const formattedDate = new Date(stats.lastUpdated).toLocaleDateString(
    undefined,
    { month: "short", day: "numeric", year: "numeric" },
  );

  return (
    <StatsBar
      items={[
        {
          label: "Datapoints",
          value: stats.totalCount.toLocaleString(),
        },
        {
          label: "Types",
          custom: (
            <div className="flex items-center gap-3">
              {stats.chatCount > 0 && (
                <div className="flex items-center gap-1.5">
                  <span className="bg-bg-type-chat rounded-sm p-0.5">
                    <TypeChat className="text-fg-type-chat" />
                  </span>
                  <span className="text-fg-primary text-sm font-medium">
                    {stats.chatCount}
                  </span>
                </div>
              )}
              {stats.jsonCount > 0 && (
                <div className="flex items-center gap-1.5">
                  <span className="bg-bg-type-json rounded-sm p-0.5">
                    <TypeJson className="text-fg-type-json" />
                  </span>
                  <span className="text-fg-primary text-sm font-medium">
                    {stats.jsonCount}
                  </span>
                </div>
              )}
            </div>
          ),
        },
        {
          label: "Functions",
          value: String(stats.functionBreakdown.length),
        },
        {
          label: "Output Coverage",
          value:
            stats.withOutput + stats.withoutOutput > 0
              ? `${Math.round((stats.withOutput / (stats.withOutput + stats.withoutOutput)) * 100)}%`
              : "—",
          detail: `${stats.withOutput} with output`,
        },
        { label: "Last Updated", value: formattedDate },
      ]}
    />
  );
}

function DatasetStatsBarSkeleton() {
  return <StatsBarSkeleton count={4} />;
}

export default function DatasetDetailPage({
  loaderData,
}: Route.ComponentProps) {
  const {
    dataset_name,
    countPromise,
    statsPromise,
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
      <PageHeader
        eyebrow={
          <Breadcrumbs segments={[{ label: "Datasets", href: "/datasets" }]} />
        }
        name={dataset_name}
        count={countPromise}
      >
        <ActionBar>
          <DeleteButton
            onClick={handleDelete}
            isLoading={fetcher.state === "submitting"}
            disabled={isReadOnly}
          />
          <AskAutopilotButton message={`Dataset: ${dataset_name}\n\n`} />
        </ActionBar>
      </PageHeader>

      <Suspense fallback={<DatasetStatsBarSkeleton />}>
        <Await resolve={statsPromise}>
          {(stats) => <DatasetStatsBar stats={stats} />}
        </Await>
      </Suspense>

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
