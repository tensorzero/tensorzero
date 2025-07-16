import {
  useSearchParams,
  useNavigate,
  data,
  Await,
  useFetcher,
  Link,
} from "react-router";
import { DatasetSelector } from "~/components/dataset/DatasetSelector";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { PageHeader, PageLayout } from "~/components/layout/PageLayout";
import { useConfig } from "~/context/config";
import { getConfig } from "~/utils/config/index.server";
import type { Route } from "./+types/route";
import { getTensorZeroClient, listDatapoints } from "~/utils/tensorzero.server";
import { VariantFilter } from "~/components/function/variant/variant-filter";
import {
  prepareInferenceActionRequest,
  tensorZeroResolvedInputToInput,
} from "~/routes/api/tensorzero/inference.utils";
import { resolveInput } from "~/utils/resolve.server";
import { Loader2 } from "lucide-react";
import type {
  FunctionConfig,
  Datapoint as TensorZeroDatapoint,
} from "tensorzero-node";
import type { DisplayInput } from "~/utils/clickhouse/common";
import { Suspense } from "react";
import {
  InferenceRequestSchema,
  type InferenceResponse,
} from "~/utils/tensorzero";
import NewOutput from "~/components/inference/NewOutput";
import { Refresh } from "~/components/icons/Icons";
import { Button } from "~/components/ui/button";
import PageButtons from "~/components/utils/PageButtons";
import { countDatapointsForDatasetFunction } from "~/utils/clickhouse/datasets.server";
import InputSnippet from "~/components/inference/InputSnippet";
import { Label } from "~/components/ui/label";

const DEFAULT_LIMIT = 10;

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const functionName = searchParams.get("functionName");
  const limit = searchParams.get("limit")
    ? parseInt(searchParams.get("limit")!)
    : DEFAULT_LIMIT;
  const offset = searchParams.get("offset")
    ? parseInt(searchParams.get("offset")!)
    : 0;
  const refreshVariantName = searchParams.get("refreshVariantName");
  const refreshDatapointId = searchParams.get("refreshDatapointId");
  const config = await getConfig();
  const functionConfig = functionName
    ? (config.functions[functionName] ?? null)
    : null;
  if (functionName && !functionConfig) {
    throw data(`Function config not found for function ${functionName}`, {
      status: 404,
    });
  }
  const datasetName = searchParams.get("datasetName");
  const selectedVariants = searchParams.getAll("variant");
  const [datapoints, totalDatapoints] = datasetName
    ? await Promise.all([
        listDatapoints(datasetName, functionName ?? undefined, limit, offset),
        functionName
          ? countDatapointsForDatasetFunction(datasetName, functionName)
          : null,
      ])
    : [undefined, null];
  // If we're refreshing a specific datapoint/variant, we should short-circuit the loader
  // and return the inference result
  if (
    refreshDatapointId &&
    refreshVariantName &&
    functionName &&
    functionConfig
  ) {
    const datapoint = datapoints?.find(
      (datapoint) => datapoint.id === refreshDatapointId,
    );
    if (!datapoint) {
      throw data(`Datapoint not found for id ${refreshDatapointId}`, {
        status: 404,
      });
    }
    return refreshInference(
      datapoint,
      functionName,
      functionConfig,
      refreshVariantName,
    );
  }
  const inputs = datapoints
    ? await Promise.all(
        datapoints.map(async (datapoint) => {
          const inputData = tensorZeroResolvedInputToInput(datapoint.input);
          return await resolveInput(inputData, functionConfig ?? null);
        }),
      )
    : undefined;

  // Create a closure we can apply to each datapoint x variant pair
  // and return the promises from the loader
  const serverInference = async (
    input: DisplayInput,
    datapoint: TensorZeroDatapoint,
    functionName: string,
    variantName: string,
  ) => {
    const request = prepareInferenceActionRequest({
      source: "clickhouse_datapoint",
      input,
      functionName,
      variant: variantName,
      tool_params:
        datapoint?.type === "chat"
          ? (datapoint.tool_params ?? undefined)
          : undefined,
      output_schema:
        datapoint?.type === "json" ? datapoint.output_schema : null,
    });
    const result = InferenceRequestSchema.safeParse(request);
    if (!result.success) {
      throw new Error("Invalid request");
    }
    return await getTensorZeroClient().inference({
      ...result.data,
      stream: false,
    });
  };
  // Do not block on all the server inferences, just return the promises
  // Create a flat array of promises, one for each datapoint/variant combination
  const serverInferences =
    functionName && datapoints && inputs
      ? datapoints.flatMap((datapoint, index) =>
          selectedVariants.map((variant) =>
            serverInference(inputs[index], datapoint, functionName, variant),
          ),
        )
      : undefined;

  return {
    type: "pageLoad" as const,
    functionName,
    datasetName,
    datapoints,
    inputs,
    serverInferences,
    totalDatapoints,
    offset,
    limit,
  };
}

async function refreshInference(
  datapoint: TensorZeroDatapoint,
  functionName: string,
  functionConfig: FunctionConfig,
  variantName: string,
) {
  const inputData = tensorZeroResolvedInputToInput(datapoint.input);
  const displayInput = await resolveInput(inputData, functionConfig ?? null);

  const request = prepareInferenceActionRequest({
    source: "clickhouse_datapoint",
    input: displayInput,
    functionName,
    variant: variantName,
    tool_params:
      datapoint?.type === "chat"
        ? (datapoint.tool_params ?? undefined)
        : undefined,
    output_schema: datapoint?.type === "json" ? datapoint.output_schema : null,
  });
  const result = InferenceRequestSchema.safeParse(request);
  if (!result.success) {
    throw new Error("Invalid request");
  }
  return {
    type: "refreshInference" as const,
    inference: await getTensorZeroClient().inference({
      ...result.data,
      stream: false,
    }),
  };
}

export default function PlaygroundPage({ loaderData }: Route.ComponentProps) {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const config = useConfig();
  // Handle refresh response differently
  if (loaderData.type === "refreshInference") {
    // This shouldn't happen at the page level, only in the fetcher
    return null;
  }

  const {
    functionName,
    datasetName,
    datapoints,
    inputs,
    serverInferences,
    totalDatapoints,
    offset,
    limit,
  } = loaderData;

  // Get selected variants from search params
  const selectedVariants = searchParams.getAll("variant");
  const functionConfig = functionName ? config.functions[functionName] : null;
  if (functionName && !functionConfig) {
    throw data(`Function config not found for function ${functionName}`, {
      status: 404,
    });
  }
  const variants = functionConfig?.variants ?? undefined;
  const variantData = variants
    ? Object.entries(variants).map(([variantName]) => ({
        name: variantName,
        color: undefined,
      }))
    : [];

  const updateSearchParams = (
    updates: Record<string, string | string[] | null>,
  ) => {
    const newParams = new URLSearchParams(searchParams);

    Object.entries(updates).forEach(([key, value]) => {
      if (value === null) {
        newParams.delete(key);
      } else if (Array.isArray(value)) {
        // Remove all existing params with this key
        newParams.delete(key);
        // Add each value in the array
        value.forEach((v) => newParams.append(key, v));
      } else {
        newParams.set(key, value);
      }
    });

    navigate(`?${newParams.toString()}`, { replace: true });
  };

  return (
    <PageLayout>
      <PageHeader name="Playground" />
      <Label>Function</Label>
      <FunctionSelector
        selected={functionName}
        onSelect={(value) =>
          updateSearchParams({ functionName: value, variant: null })
        }
        functions={config.functions}
      />
      <Label>Dataset</Label>
      <DatasetSelector
        selected={datasetName ?? undefined}
        onSelect={(value) => updateSearchParams({ datasetName: value })}
        allowCreation={false}
      />
      <Label>Variants</Label>
      <VariantFilter
        disabled={!functionName || !datasetName}
        variants={variantData}
        selectedValues={selectedVariants}
        setSelectedValues={(valuesOrUpdater) => {
          const newValues =
            typeof valuesOrUpdater === "function"
              ? valuesOrUpdater(selectedVariants)
              : valuesOrUpdater;
          updateSearchParams({ variant: newValues });
        }}
      />
      {selectedVariants.length > 0 &&
        datapoints &&
        datapoints.length > 0 &&
        datasetName &&
        inputs &&
        functionName && (
          <div className="-mx-4 mt-6 md:-mx-8 lg:-mx-16">
            <div
              className="overflow-x-auto overflow-y-auto px-4"
              style={{ maxHeight: "calc(100vh - 200px)" }}
            >
              <div className="min-w-fit">
                {/* Header row with sticky positioning */}
                <div className="bg-background sticky top-0 z-20 grid grid-cols-[400px_1fr] border-b">
                  <div className="bg-background sticky left-0 z-30 border-r p-4 font-medium">
                    Datapoint Input
                  </div>
                  <div className="grid auto-cols-[minmax(320px,1fr)] grid-flow-col">
                    {selectedVariants.map((variant) => (
                      <div
                        key={variant}
                        className="border-r p-4 font-medium last:border-r-0"
                      >
                        <Link
                          to={`/observability/functions/${encodeURIComponent(functionName)}/variants/${encodeURIComponent(variant)}`}
                          className="text-blue-600 hover:text-blue-800 hover:underline"
                        >
                          {variant}
                        </Link>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Data rows */}
                {datapoints.map(
                  (datapoint: TensorZeroDatapoint, index: number) => (
                    <div
                      key={datapoint.id}
                      className="grid grid-cols-[400px_1fr] border-b"
                    >
                      <div className="bg-background sticky left-0 z-10 border-r p-4 font-mono text-sm">
                        <div className="text-xs text-gray-500">
                          Datapoint:{" "}
                          <Link
                            to={`/datasets/${encodeURIComponent(datasetName)}/datapoint/${datapoint.id}`}
                            className="text-blue-600 hover:text-blue-800 hover:underline"
                          >
                            {datapoint.id}
                          </Link>
                        </div>
                        <InputSnippet
                          messages={inputs[index].messages}
                          system={inputs[index].system}
                        />
                      </div>
                      <div className="grid auto-cols-[minmax(320px,1fr)] grid-flow-col">
                        {selectedVariants.map((variant, variantIndex) => {
                          const inferenceIndex =
                            index * selectedVariants.length + variantIndex;
                          return (
                            <div
                              key={`${datapoint.id}-${variant}`}
                              className="border-r p-4 last:border-r-0"
                            >
                              <DatapointPlaygroundOutput
                                datapoint={datapoint}
                                variantName={variant}
                                serverInference={
                                  serverInferences?.[inferenceIndex]
                                }
                              />
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ),
                )}
              </div>
            </div>
          </div>
        )}
      <PageButtons
        onPreviousPage={() => {
          const newOffset = Math.max(0, offset - limit);
          updateSearchParams({ offset: newOffset.toString() });
        }}
        onNextPage={() => {
          const newOffset = offset + limit;
          updateSearchParams({ offset: newOffset.toString() });
        }}
        disablePrevious={offset === 0}
        disableNext={
          totalDatapoints ? offset + limit >= totalDatapoints : false
        }
      />
    </PageLayout>
  );
}

interface DatapointPlaygroundOutputProps {
  datapoint: TensorZeroDatapoint;
  variantName: string;
  serverInference: Promise<InferenceResponse> | undefined;
}
function DatapointPlaygroundOutput({
  datapoint,
  variantName,
  serverInference,
}: DatapointPlaygroundOutputProps) {
  const fetcher = useFetcher<typeof loader>();

  // Check if currently refreshing
  const isRefreshing = fetcher.state !== "idle";

  // Get the refreshed data if available
  const refreshedInference =
    fetcher.data?.type === "refreshInference" ? fetcher.data.inference : null;

  if (!serverInference && !refreshedInference) {
    return (
      <div className="flex min-h-[8rem] items-center justify-center">
        <div className="text-muted-foreground text-sm">
          No inference available
        </div>
      </div>
    );
  }

  // Show loading state when refreshing
  if (isRefreshing) {
    return (
      <div className="group relative">
        <div className="flex min-h-[8rem] items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin" />
        </div>
      </div>
    );
  }

  return (
    <div className="group relative">
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-1 left-1 z-10 opacity-0 transition-opacity group-hover:opacity-100"
        onClick={() => {
          const url = new URL(window.location.href);
          const queryParams = new URLSearchParams(url.search);
          // Set the refresh params
          queryParams.set("refreshDatapointId", datapoint.id);
          queryParams.set("refreshVariantName", variantName);
          fetcher.load(`/playground?${queryParams.toString()}`);
        }}
      >
        <Refresh />
      </Button>

      {refreshedInference ? (
        // Display refreshed data directly
        <NewOutput
          output={
            "content" in refreshedInference
              ? refreshedInference.content
              : refreshedInference.output
          }
        />
      ) : (
        // Display initial server inference with Suspense
        <Suspense
          fallback={
            <div className="flex min-h-[8rem] items-center justify-center">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
          }
        >
          <Await
            resolve={serverInference}
            errorElement={
              <div className="flex min-h-[8rem] items-center justify-center">
                <div className="text-center text-red-600">
                  <p className="font-semibold">Error</p>
                  <p className="text-sm">Failed to load inference</p>
                </div>
              </div>
            }
          >
            {(response) => {
              if (!response) {
                return (
                  <div className="flex min-h-[8rem] items-center justify-center">
                    <div className="text-muted-foreground text-sm">
                      No response available
                    </div>
                  </div>
                );
              }
              let output;
              if ("content" in response) {
                output = response.content;
              } else {
                output = response.output;
              }
              return <NewOutput output={output} />;
            }}
          </Await>
        </Suspense>
      )}
    </div>
  );
}
