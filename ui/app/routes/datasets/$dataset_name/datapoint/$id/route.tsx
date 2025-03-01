import { useFetcher, Link } from "react-router";
import { data, isRouteErrorResponse, redirect } from "react-router";
import BasicInfo from "./BasicInfo";
import Input from "~/components/inference/Input";
import Output from "~/components/inference/Output";
import { useState } from "react";
import { useConfig } from "~/context/config";
import { getDatapoint } from "~/utils/clickhouse/datasets.server";
import { VariantResponseModal } from "./VariantResponseModal";
import type { Route } from "./+types/route";
import type { ActionFunctionArgs } from "react-router";
import {
  ParsedDatasetRowSchema,
  type ParsedDatasetRow,
} from "~/utils/clickhouse/datasets";
import {
  deleteDatapoint as deleteDatapointServer,
  getDatasetCounts,
} from "~/utils/clickhouse/datasets.server";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";

export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();

  const rawData = {
    dataset_name: formData.get("dataset_name"),
    function_name: formData.get("function_name"),
    id: formData.get("id"),
    episode_id: formData.get("episode_id"),
    input: JSON.parse(formData.get("input") as string),
    output: formData.get("output")
      ? JSON.parse(formData.get("output") as string)
      : undefined,
    output_schema: formData.get("output_schema")
      ? JSON.parse(formData.get("output_schema") as string)
      : undefined,
    tags: JSON.parse(formData.get("tags") as string),
    auxiliary: formData.get("auxiliary"),
    is_deleted: formData.get("is_deleted") === "true",
    updated_at: formData.get("updated_at"),
  };

  const parsedFormData: ParsedDatasetRow =
    ParsedDatasetRowSchema.parse(rawData);
  await deleteDatapointServer(parsedFormData);
  const datasetCounts = await getDatasetCounts();
  const datasetCount = datasetCounts.find(
    (count) => count.dataset_name === parsedFormData.dataset_name,
  );

  if (datasetCount === undefined) {
    return redirect("/datasets");
  }
  return redirect(`/datasets/${parsedFormData.dataset_name}`);
}

export async function loader({
  params,
}: {
  params: { dataset_name: string; id: string };
}) {
  const { dataset_name, id } = params;
  if (!dataset_name || !id) {
    throw data(`No datapoint found for id ${id}.`, {
      status: 404,
    });
  }
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

  const deleteFetcher = useFetcher();
  const handleDelete = () => {
    const formData = new FormData();
    Object.entries(datapoint).forEach(([key, value]) => {
      if (value === undefined) return;
      if (value === null) {
        formData.append(key, "null");
      } else if (typeof value === "object") {
        formData.append(key, JSON.stringify(value));
      } else {
        formData.append(key, String(value));
      }
    });

    // Submit to the local action by targeting the current route (".")
    deleteFetcher.submit(formData, { method: "post", action: "." });
  };

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
    <div className="container mx-auto px-4 pb-8">
      <PageLayout>
        <div className="flex flex-col gap-3">
          <PageHeader heading="Datapoint" name={datapoint.id} />
          <div className="text-sm text-foreground">
            Dataset{" "}
            <Link
              to={`/datasets/${datapoint.dataset_name}`}
              className="rounded bg-background-tertiary px-1.5 py-1 font-mono font-semibold"
            >
              {datapoint.dataset_name}
            </Link>
          </div>
        </div>

        <SectionsGroup>
          <SectionLayout>
            <BasicInfo
              datapoint={datapoint}
              tryWithVariantProps={{
                variants,
                onVariantSelect,
                isLoading: variantInferenceIsLoading,
              }}
              onDelete={handleDelete}
              isDeleting={deleteFetcher.state === "submitting"}
            />
          </SectionLayout>

          <SectionLayout>
            <SectionHeader heading="Input" />
            <Input input={datapoint.input} />
          </SectionLayout>

          {datapoint.output && (
            <SectionLayout>
              <SectionHeader heading="Output" />
              <Output output={datapoint.output} />
            </SectionLayout>
          )}
        </SectionsGroup>

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
      </PageLayout>
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
