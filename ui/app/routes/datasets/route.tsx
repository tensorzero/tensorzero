import type { Route } from "./+types/route";
import DatasetTable from "./DatasetTable";
import { data, isRouteErrorResponse } from "react-router";
import { useNavigate } from "react-router";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { DatasetsActions } from "./DatasetsActions";
import { logger } from "~/utils/logger";
import { getNativeTensorZeroClient } from "~/utils/tensorzero/native_client.server";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export async function loader(_args: Route.LoaderArgs) {
  const datasetMetadata = await getTensorZeroClient().listDatasets({});
  return { datasets: datasetMetadata.datasets };
}

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();
  const action = formData.get("action");
  if (action === "delete") {
    const datasetName = formData.get("datasetName");
    if (typeof datasetName !== "string") {
      throw data("Dataset name is required", { status: 400 });
    }
    const client = await getNativeTensorZeroClient();
    const staleDataset = await client.staleDataset(datasetName);
    return staleDataset;
  }
  return null;
}

export default function DatasetListPage({ loaderData }: Route.ComponentProps) {
  const { datasets } = loaderData;
  const navigate = useNavigate();
  return (
    <PageLayout>
      <PageHeader heading="Datasets" count={datasets.length} />
      <SectionLayout>
        <DatasetsActions onBuildDataset={() => navigate("/datasets/builder")} />
        <DatasetTable datasets={datasets} />
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
