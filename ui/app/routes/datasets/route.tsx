import type { Route } from "./+types/route";
import DatasetTable from "./DatasetTable";
import { data } from "react-router";
import { Await, useNavigate } from "react-router";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { DatasetsActions } from "./DatasetsActions";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { DatasetMetadata } from "~/types/tensorzero";
import { Suspense } from "react";
import { StatsBar, StatsBarSkeleton } from "~/components/ui/StatsBar";

export async function loader() {
  const dataPromise = getTensorZeroClient()
    .listDatasets({})
    .then((r) => r.datasets);
  const countPromise = dataPromise.then((d) => d.length);

  return {
    dataPromise,
    countPromise,
  };
}

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();
  const action = formData.get("action");
  if (action === "delete") {
    const datasetName = formData.get("datasetName");
    if (typeof datasetName !== "string") {
      throw data("Dataset name is required", { status: 400 });
    }
    const client = await getTensorZeroClient();
    const deleteDatasetResponse = await client.deleteDataset(datasetName);
    return deleteDatasetResponse;
  }
  return null;
}

function DatasetsSummary({ datasets }: { datasets: DatasetMetadata[] }) {
  if (datasets.length === 0) return null;

  const totalDatapoints = datasets.reduce(
    (sum, d) => sum + d.datapoint_count,
    0,
  );
  const lastUpdated = datasets.reduce((latest, d) => {
    const t = new Date(d.last_updated).getTime();
    return t > latest ? t : latest;
  }, 0);
  const formattedDate = new Date(lastUpdated).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });

  return (
    <StatsBar
      items={[
        { label: "Datasets", value: String(datasets.length) },
        {
          label: "Total Datapoints",
          value: totalDatapoints.toLocaleString(),
        },
        {
          label: "Avg per Dataset",
          value: Math.round(totalDatapoints / datasets.length).toLocaleString(),
        },
        { label: "Last Updated", value: formattedDate },
      ]}
    />
  );
}

function DatasetsSummarySkeleton() {
  return <StatsBarSkeleton count={4} />;
}

export default function DatasetListPage({ loaderData }: Route.ComponentProps) {
  const { dataPromise, countPromise } = loaderData;
  const navigate = useNavigate();
  return (
    <PageLayout>
      <PageHeader heading="Datasets" count={countPromise} />
      <Suspense fallback={<DatasetsSummarySkeleton />}>
        <Await resolve={dataPromise}>
          {(datasets) => <DatasetsSummary datasets={datasets} />}
        </Await>
      </Suspense>
      <SectionLayout>
        <DatasetsActions
          onBuildDataset={() => navigate("/datasets/builder")}
          onNewDatapoint={() => navigate("/datapoints/new")}
        />
        <DatasetTable data={dataPromise} />
      </SectionLayout>
    </PageLayout>
  );
}
