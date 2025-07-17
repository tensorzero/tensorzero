import {
  useSearchParams,
  useNavigate,
  data,
  Await,
  Link,
  useAsyncError,
  type RouteHandle,
  type ShouldRevalidateFunctionArgs,
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
import { Loader2, X } from "lucide-react";
import type { Datapoint as TensorZeroDatapoint } from "tensorzero-node";
import type { DisplayInput } from "~/utils/clickhouse/common";
import { Suspense, useCallback, useEffect, useState } from "react";
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
import { CodeEditor } from "~/components/ui/code-editor";

const DEFAULT_LIMIT = 10;

export const handle: RouteHandle = {
  crumb: () => ["Playground"],
};

/**
 * We will skip revalidation on navigation in the case where:
 * - The previous route was the same as the current route
 * - The previous route shared the same functionName, datasetName, limit, and offset
 */
export function shouldRevalidate(arg: ShouldRevalidateFunctionArgs) {
  const { currentUrl, nextUrl } = arg;
  // First check that the base route is the same
  if (currentUrl.pathname !== nextUrl.pathname) {
    return true;
  }
  // Then check that the search params are the same
  const currentSearchParams = new URLSearchParams(currentUrl.search);
  const nextSearchParams = new URLSearchParams(nextUrl.search);
  const currentFunctionName = currentSearchParams.get("functionName");
  const nextFunctionName = nextSearchParams.get("functionName");
  const currentDatasetName = currentSearchParams.get("datasetName");
  const nextDatasetName = nextSearchParams.get("datasetName");
  const currentLimit = currentSearchParams.get("limit");
  const nextLimit = nextSearchParams.get("limit");
  const currentOffset = currentSearchParams.get("offset");
  const nextOffset = nextSearchParams.get("offset");
  if (
    currentFunctionName === nextFunctionName &&
    currentDatasetName === nextDatasetName &&
    currentLimit === nextLimit &&
    currentOffset === nextOffset
  ) {
    return false;
  }
  return true;
}

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
  // const refreshVariantName = searchParams.get("refreshVariantName");
  // const refreshDatapointId = searchParams.get("refreshDatapointId");
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
  /*
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
    try {
      return await refreshInference(
        datapoint,
        functionName,
        functionConfig,
        refreshVariantName,
      );
    } catch (error) {
      // Return error as part of the response instead of throwing
      return {
        type: "refreshInferenceError" as const,
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
      };
    }
  }*/
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
  // Create a map of maps of promises, one for each datapoint/variant combination
  // The structure should be: serverInferences[variantName][datapointId] = promise
  // We can use this to avoid re-running the same inference multiple times
  const serverInferences = new Map<
    string,
    Map<string, Promise<InferenceResponse>>
  >();
  for (const variant of selectedVariants) {
    serverInferences.set(variant, new Map());
  }
  if (datapoints && inputs && functionName) {
    for (let index = 0; index < datapoints.length; index++) {
      const datapoint = datapoints[index];
      const input = inputs[index];
      for (const variant of selectedVariants) {
        serverInferences
          .get(variant)
          ?.set(
            datapoint.id,
            serverInference(input, datapoint, functionName, variant),
          );
      }
    }
  }

  return {
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

// async function refreshInference(
//   datapoint: TensorZeroDatapoint,
//   functionName: string,
//   functionConfig: FunctionConfig,
//   variantName: string,
// ) {
//   const inputData = tensorZeroResolvedInputToInput(datapoint.input);
//   const displayInput = await resolveInput(inputData, functionConfig ?? null);

//   const request = prepareInferenceActionRequest({
//     source: "clickhouse_datapoint",
//     input: displayInput,
//     functionName,
//     variant: variantName,
//     tool_params:
//       datapoint?.type === "chat"
//         ? (datapoint.tool_params ?? undefined)
//         : undefined,
//     output_schema: datapoint?.type === "json" ? datapoint.output_schema : null,
//   });
//   const result = InferenceRequestSchema.safeParse(request);
//   if (!result.success) {
//     throw new Error("Invalid request");
//   }
//   return {
//     type: "refreshInference" as const,
//     inference: await getTensorZeroClient().inference({
//       ...result.data,
//       stream: false,
//     }),
//   };
// }

export default function PlaygroundPage({ loaderData }: Route.ComponentProps) {
  const [searchParams] = useSearchParams();
  const selectedVariants = searchParams.getAll("variant");
  const navigate = useNavigate();
  const config = useConfig();
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
  const { map, setPromise } = useClientInferences(
    functionName,
    datapoints,
    inputs,
    selectedVariants,
    serverInferences,
  );

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
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => {
                            updateSearchParams({
                              variant: selectedVariants.filter(
                                (v) => v !== variant,
                              ),
                            });
                          }}
                        >
                          <X />
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>

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
                        {selectedVariants.map((variant) => {
                          return (
                            <div
                              key={`${datapoint.id}-${variant}`}
                              className="border-r p-4 last:border-r-0"
                            >
                              <DatapointPlaygroundOutput
                                datapoint={datapoint}
                                variantName={variant}
                                serverInference={map
                                  .get(variant)
                                  ?.get(datapoint.id)}
                                setPromise={setPromise}
                                input={inputs[index]}
                                functionName={functionName}
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
  setPromise: (
    variantName: string,
    datapointId: string,
    promise: Promise<InferenceResponse>,
  ) => void;
  input: DisplayInput;
  functionName: string;
}
function DatapointPlaygroundOutput({
  datapoint,
  variantName,
  serverInference,
  setPromise,
  input,
  functionName,
}: DatapointPlaygroundOutputProps) {
  if (!serverInference) {
    return (
      <div className="flex min-h-[8rem] items-center justify-center">
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-1 left-1 z-10 opacity-0 transition-opacity group-hover:opacity-100"
          onClick={() => {
            refreshClientInference(
              setPromise,
              input,
              datapoint,
              variantName,
              functionName,
            );
          }}
        >
          <Refresh />
        </Button>
        <div className="text-muted-foreground text-sm">
          No inference available
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
          refreshClientInference(
            setPromise,
            input,
            datapoint,
            variantName,
            functionName,
          );
        }}
      >
        <Refresh />
      </Button>
      <Suspense
        fallback={
          <div className="flex min-h-[8rem] items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin" />
          </div>
        }
      >
        <Await resolve={serverInference} errorElement={<InferenceError />}>
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
    </div>
  );
}

function InferenceError() {
  const error = useAsyncError();
  const isInferenceError = error instanceof Error;

  return (
    <div className="flex min-h-[8rem] items-center justify-center">
      <div className="max-h-[16rem] max-w-md overflow-y-auto px-4 text-center text-red-600">
        <p className="font-semibold">Error</p>
        <p className="mt-1 text-sm">
          {isInferenceError ? (
            <CodeEditor value={error.message} readOnly />
          ) : (
            "Failed to load inference"
          )}
        </p>
      </div>
    </div>
  );
}

type NestedPromiseMap<T> = Map<string, Map<string, Promise<T>>>;

function useNestedPromiseMap<T>(initialMap: NestedPromiseMap<T>) {
  const [map, setMap] = useState<NestedPromiseMap<T>>(initialMap);
  const setPromise = useCallback(
    (outerKey: string, innerKey: string, promise: Promise<T>) => {
      setMap((prevMap) => {
        const newMap = new Map(prevMap);
        const innerMap = newMap.get(outerKey) || new Map();
        const newInnerMap = new Map(innerMap);
        newInnerMap.set(innerKey, promise);
        newMap.set(outerKey, newInnerMap);
        return newMap;
      });
    },
    [],
  );
  return { map, setPromise, setMap };
}

function useClientInferences(
  functionName: string | null,
  datapoints: TensorZeroDatapoint[] | undefined,
  inputs: DisplayInput[] | undefined,
  selectedVariants: string[],
  serverInferences: NestedPromiseMap<InferenceResponse>,
) {
  const { map, setPromise, setMap } =
    useNestedPromiseMap<InferenceResponse>(serverInferences);
  useEffect(() => {
    setMap(serverInferences);
  }, [serverInferences, setMap]);
  // Effect to ensure all needed inferences are present
  useEffect(() => {
    if (!functionName || !datapoints || !inputs) return;

    // Check each required combination
    selectedVariants.forEach((variant) => {
      // Ensure the variant key exists in the map
      if (!map.has(variant)) {
        map.set(variant, new Map());
      }

      datapoints.forEach((datapoint, index) => {
        // Check if this inference already exists in the map
        const existingPromise = map.get(variant)?.get(datapoint.id);

        // If it doesn't exist, create it
        if (!existingPromise) {
          const input = inputs[index];
          refreshClientInference(
            setPromise,
            input,
            datapoint,
            variant,
            functionName,
          );
        }
      });
    });
  }, [functionName, datapoints, inputs, selectedVariants, map, setPromise]);

  return { map, setPromise, setMap };
}

function refreshClientInference(
  setPromise: (
    outerKey: string,
    innerKey: string,
    promise: Promise<InferenceResponse>,
  ) => void,
  input: DisplayInput,
  datapoint: TensorZeroDatapoint,
  variantName: string,
  functionName: string,
) {
  const request = prepareInferenceActionRequest({
    source: "clickhouse_datapoint",
    input,
    functionName,
    variant: variantName,
    tool_params:
      datapoint?.type === "chat"
        ? (datapoint.tool_params ?? undefined)
        : undefined,
    output_schema: datapoint?.type === "json" ? datapoint.output_schema : null,
  });
  // The API endpoint takes form data so we need to stringify it and send as data
  const formData = new FormData();
  formData.append("data", JSON.stringify(request));
  const responsePromise = async () => {
    const response = await fetch("/api/tensorzero/inference", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    return data;
  };
  setPromise(variantName, datapoint.id, responsePromise());
}
