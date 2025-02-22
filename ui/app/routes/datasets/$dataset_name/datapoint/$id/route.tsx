import type { Route } from "./+types/route";
import { data, isRouteErrorResponse, Link } from "react-router";
import BasicInfo from "./BasicInfo";
import Input from "~/components/inference/Input";
import Output from "~/components/inference/Output";
import { useState } from "react";
import { useConfig } from "~/context/config";
import {
  getDatapoint,
  getDatasetCounts,
} from "~/utils/clickhouse/datasets.server";
import { VariantResponseModal } from "./VariantResponseModal";
import { useDatapointDeleter } from "~/routes/api/datasets/delete.route";

export async function loader({ params }: Route.LoaderArgs) {
  const { dataset_name, id } = params;
  const datapoint = await getDatapoint(dataset_name, id);
  const datasetCounts = await getDatasetCounts();
  const datasetCount = datasetCounts.find(
    (count) => count.dataset_name === dataset_name,
  );
  if (!datapoint) {
    throw data(`No datapoint found for id ${id}.`, {
      status: 404,
    });
  }
  if (!datasetCount) {
    throw data(
      `No dataset count found for dataset ${dataset_name}. This should never happen. Please open a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.`,
      {
        status: 500,
      },
    );
  }
  console.log(datasetCount);

  return {
    datapoint,
    count: datasetCount.count,
  };
}

export default function DatapointPage({ loaderData }: Route.ComponentProps) {
  const { datapoint, count } = loaderData;
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [variantInferenceIsLoading, setVariantInferenceIsLoading] =
    useState(false);
  const [selectedVariant, setSelectedVariant] = useState<string | null>(null);
  const config = useConfig();
  const { deleteDatapoint, isDeleting, isDeleted } = useDatapointDeleter();
  const variants = Object.keys(
    config.functions[datapoint.function_name]?.variants || {},
  );
  console.log(isDeleted);

  const onVariantSelect = (variant: string) => {
    setSelectedVariant(variant);
    setIsModalOpen(true);
  };

  const handleModalClose = () => {
    setIsModalOpen(false);
    setSelectedVariant(null);
    setVariantInferenceIsLoading(false);
  };

  const handleDelete = () => {
    deleteDatapoint(datapoint, count);
  };

  return (
    <div className="container mx-auto space-y-6 p-4">
      <h2 className="mb-4 text-2xl font-semibold">
        Datapoint{" "}
        <code className="rounded bg-gray-100 p-1 text-2xl">{datapoint.id}</code>
        in dataset{" "}
        <Link to={`/datasets/${datapoint.dataset_name}`}>
          <code className="rounded bg-gray-100 p-1 text-2xl">
            {datapoint.dataset_name}
          </code>
        </Link>
      </h2>
      <div className="mb-6 h-px w-full bg-gray-200"></div>

      <BasicInfo
        datapoint={datapoint}
        tryWithVariantProps={{
          variants,
          onVariantSelect,
          isLoading: variantInferenceIsLoading,
        }}
        onDelete={handleDelete}
        isDeleting={isDeleting}
      />
      <Input input={datapoint.input} />
      {datapoint.output && <Output output={datapoint.output} />}
      {selectedVariant && (
        <VariantResponseModal
          isOpen={isModalOpen}
          isLoading={variantInferenceIsLoading}
          setIsLoading={setVariantInferenceIsLoading}
          onClose={handleModalClose}
          datapoint={datapoint}
          selectedVariant={selectedVariant}
        />
      )}
    </div>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  console.error(error);

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
