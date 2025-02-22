import type { Route } from "./+types/route";
import { data, isRouteErrorResponse } from "react-router";
import BasicInfo from "./BasicInfo";
import Input from "~/components/inference/Input";
import Output from "~/components/inference/Output";
import { useState } from "react";
import { useConfig } from "~/context/config";
import { getDatapoint } from "~/utils/clickhouse/datasets.server";
import { VariantResponseModal } from "./VariantResponseModal";

export async function loader({ params }: Route.LoaderArgs) {
  const { dataset_name, id } = params;
  const datapoint = await getDatapoint(dataset_name, id);
  if (!datapoint) {
    throw data(`No datapoint found for id ${id}.`, {
      status: 404,
    });
  }

  return {
    datapoint,
  };
}

export default function DatapointPage({ loaderData }: Route.ComponentProps) {
  const { datapoint } = loaderData;
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [variantInferenceIsLoading, setVariantInferenceIsLoading] =
    useState(false);
  const [selectedVariant, setSelectedVariant] = useState<string | null>(null);
  const config = useConfig();
  const variants = Object.keys(
    config.functions[datapoint.function_name]?.variants || {},
  );

  const onVariantSelect = (variant: string) => {
    setSelectedVariant(variant);
    setIsModalOpen(true);
  };

  const handleModalClose = () => {
    setIsModalOpen(false);
    setSelectedVariant(null);
    setVariantInferenceIsLoading(false);
  };

  return (
    <div className="container mx-auto space-y-6 p-4">
      <h2 className="mb-4 text-2xl font-semibold">
        Datapoint{" "}
        <code className="rounded bg-gray-100 p-1 text-2xl">{datapoint.id}</code>
      </h2>
      <div className="mb-6 h-px w-full bg-gray-200"></div>

      <BasicInfo
        datapoint={datapoint}
        tryWithVariantProps={{
          variants,
          onVariantSelect,
          isLoading: variantInferenceIsLoading,
        }}
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
