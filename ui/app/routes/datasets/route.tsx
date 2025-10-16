import {
  getDatasetMetadata,
  countDatasets,
} from "~/utils/clickhouse/datasets.server";
import type { Route } from "./+types/route";
import DatasetTable from "./DatasetTable";
import { data, isRouteErrorResponse } from "react-router";
import { useNavigate } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import { DatasetsActions } from "./DatasetsActions";
import { logger } from "~/utils/logger";
import { getNativeTensorZeroClient } from "~/utils/tensorzero/native_client.server";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const pageSize = Number(url.searchParams.get("pageSize")) || 15;
  const offset = Number(url.searchParams.get("offset")) || 0;
  if (pageSize > 100) {
    throw data("Page size cannot exceed 100", { status: 400 });
  }
  const datasetMetadata = await getDatasetMetadata({
    page_size: pageSize,
    offset,
  });
  const numberOfDatasets = await countDatasets();
  return { counts: datasetMetadata, pageSize, offset, numberOfDatasets };
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
  const { counts, pageSize, offset, numberOfDatasets } = loaderData;
  const navigate = useNavigate();
  const handleNextPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset + pageSize));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };
  const handlePreviousPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset - pageSize));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };
  return (
    <PageLayout>
      <PageHeader heading="Datasets" count={numberOfDatasets} />
      <SectionLayout>
        <DatasetsActions onBuildDataset={() => navigate("/datasets/builder")} />
        <DatasetTable counts={counts} />
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={offset === 0}
          disableNext={offset + pageSize >= numberOfDatasets}
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
