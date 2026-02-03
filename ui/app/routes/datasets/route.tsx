import type { Route } from "./+types/route";
import DatasetTable from "./DatasetTable";
import { data, useNavigate } from "react-router";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { DatasetsActions } from "./DatasetsActions";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { LayoutErrorBoundary } from "~/components/ui/error";

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

export default function DatasetListPage({ loaderData }: Route.ComponentProps) {
  const { dataPromise, countPromise } = loaderData;
  const navigate = useNavigate();
  return (
    <PageLayout>
      <PageHeader heading="Datasets" count={countPromise} />
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

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  return <LayoutErrorBoundary error={error} />;
}
