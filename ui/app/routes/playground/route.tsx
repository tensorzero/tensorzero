import {
  useSearchParams,
  data,
  Link,
  type RouteHandle,
  type ShouldRevalidateFunctionArgs,
  isRouteErrorResponse,
  useNavigation,
} from "react-router";
import { DatasetSelector } from "~/components/dataset/DatasetSelector";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { PageHeader, PageLayout } from "~/components/layout/PageLayout";
import { useFunctionConfig, useAllFunctionConfigs } from "~/context/config";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import type { Route } from "./+types/route";
import { listDatapoints } from "~/utils/tensorzero.server";
import { tensorZeroResolvedInputToInput } from "~/routes/api/tensorzero/inference.utils";
import { resolveInput } from "~/utils/resolve.server";
import { X } from "lucide-react";
import type {
  FunctionConfig,
  Datapoint as TensorZeroDatapoint,
} from "tensorzero-node";
import type { DisplayInput } from "~/utils/clickhouse/common";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "~/components/ui/button";
import PageButtons from "~/components/utils/PageButtons";
import { countDatapointsForDatasetFunction } from "~/utils/clickhouse/datasets.server";
import InputSnippet from "~/components/inference/InputSnippet";
import { Output } from "~/components/inference/Output";
import { Label } from "~/components/ui/label";
import DatapointPlaygroundOutput from "./DatapointPlaygroundOutput";
import { safeParseInt } from "~/utils/common";
import { getNativeTensorZeroClient } from "~/utils/tensorzero/native_client.server";
import type { InferenceResponse } from "tensorzero-node";
import { EditButton } from "~/components/utils/EditButton";
import { VariantEditor } from "~/components/function/variant/VariantEditor";
import { Badge } from "~/components/ui/badge";
import {
  extractOriginalVariantNameFromEdited,
  getNewVariantName,
  getVariants,
  preparePlaygroundInferenceRequest,
  type PlaygroundVariantInfo,
} from "./utils";
import { BuiltinVariantFilter } from "./BuiltInVariantSelector";

const DEFAULT_LIMIT = 5;

export const handle: RouteHandle = {
  crumb: () => ["Playground"],
};

function getCleanVariantName(variant: PlaygroundVariantInfo) {
  if (variant.type === "builtin") {
    return variant.name;
  } else if (variant.type === "edited") {
    const originalVariantName = extractOriginalVariantNameFromEdited(
      variant.name,
    );
    return originalVariantName;
  }
}

function getDisplayVariantName(variant: PlaygroundVariantInfo) {
  if (variant.type === "builtin") {
    return <span>{variant.name}</span>;
  } else if (variant.type === "edited") {
    const originalVariantName = extractOriginalVariantNameFromEdited(
      variant.name,
    );
    return (
      <span>
        {originalVariantName} <Badge variant="secondary">edited</Badge>
      </span>
    );
  }
}

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
  const currentLimit = safeParseInt(
    currentSearchParams.get("limit"),
    DEFAULT_LIMIT,
  );
  const nextLimit = safeParseInt(nextSearchParams.get("limit"), DEFAULT_LIMIT);
  const currentOffset = safeParseInt(currentSearchParams.get("offset"), 0);
  const nextOffset = safeParseInt(nextSearchParams.get("offset"), 0);
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
  const limit = safeParseInt(searchParams.get("limit"), DEFAULT_LIMIT);
  const offset = safeParseInt(searchParams.get("offset"), 0);

  let config;
  try {
    config = await getConfig();
  } catch {
    throw data("Failed to load configuration", { status: 500 });
  }

  const functionConfig = functionName
    ? await getFunctionConfig(functionName, config)
    : null;
  if (functionName && !functionConfig) {
    throw data(`Function config not found for function ${functionName}`, {
      status: 404,
    });
  }
  const datasetName = searchParams.get("datasetName");
  const variants = getVariants(searchParams);

  let datapoints, totalDatapoints;
  try {
    [datapoints, totalDatapoints] = datasetName
      ? await Promise.all([
          listDatapoints(datasetName, functionName ?? undefined, limit, offset),
          functionName
            ? countDatapointsForDatasetFunction(datasetName, functionName)
            : null,
        ])
      : [undefined, null];
  } catch (error) {
    throw data(
      `Failed to load datapoints: ${error instanceof Error ? error.message : "Unknown error"}`,
      {
        status: 500,
      },
    );
  }

  let inputs;
  try {
    inputs = datapoints
      ? await Promise.all(
          datapoints.map(async (datapoint) => {
            const inputData = tensorZeroResolvedInputToInput(datapoint.input);
            return await resolveInput(inputData, functionConfig ?? null);
          }),
        )
      : undefined;
  } catch (error) {
    throw data(
      `Failed to resolve inputs: ${error instanceof Error ? error.message : "Unknown error"}`,
      {
        status: 500,
      },
    );
  }

  // Create a closure we can apply to each datapoint x variant pair
  // and return the promises from the loader
  const serverInference = async (
    input: DisplayInput,
    datapoint: TensorZeroDatapoint,
    functionName: string,
    variantInfo: PlaygroundVariantInfo,
    functionConfig: FunctionConfig,
  ) => {
    const request = preparePlaygroundInferenceRequest(
      variantInfo,
      functionName,
      datapoint,
      input,
      functionConfig,
    );
    const nativeClient = await getNativeTensorZeroClient();
    const inferenceResponse = await nativeClient.inference(request);
    return inferenceResponse;
  };
  // Do not block on all the server inferences, just return the promises
  // Create a map of maps of promises, one for each datapoint/variant combination
  // The structure should be: serverInferences[variantName][datapointId] = promise
  // We can use this to avoid re-running the same inference multiple times
  const serverInferences = new Map<
    string,
    Map<string, Promise<InferenceResponse>>
  >();
  for (const variant of variants) {
    serverInferences.set(variant.name, new Map());
  }
  if (datapoints && inputs && functionName && functionConfig) {
    for (let index = 0; index < datapoints.length; index++) {
      const datapoint = datapoints[index];
      const input = inputs[index];
      for (const variant of variants) {
        serverInferences
          .get(variant.name)
          ?.set(
            datapoint.id,
            serverInference(
              input,
              datapoint,
              functionName,
              variant,
              functionConfig,
            ),
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

export default function PlaygroundPage({ loaderData }: Route.ComponentProps) {
  const navigation = useNavigation();
  const [currentSearchParams, setSearchParams] = useSearchParams();
  const [editingVariant, setEditingVariant] =
    useState<PlaygroundVariantInfo | null>(null);
  const { searchParams, loadingVariants } = useMemo(() => {
    if (navigation.state !== "loading") {
      return {
        searchParams: currentSearchParams,
        loadingVariants: new Set<string>(),
      };
    }

    const nextSearchParams = new URLSearchParams(navigation.location?.search);
    // TODO: this is wrong
    const currentVariants = getVariants(currentSearchParams);
    const currentVariantNames = new Set<string>(
      currentVariants.map((variant) => variant.name),
    );
    const nextVariants = getVariants(nextSearchParams);
    const loadingVariants = new Set<string>();
    for (const variant of nextVariants) {
      if (!currentVariantNames.has(variant.name)) {
        loadingVariants.add(variant.name);
      }
    }

    return {
      searchParams: nextSearchParams,
      loadingVariants,
    };
  }, [navigation, currentSearchParams]);
  const variants = getVariants(searchParams);

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
  const functionConfig = useFunctionConfig(functionName);
  if (functionName && !functionConfig) {
    throw data(`Function config not found for function ${functionName}`, {
      status: 404,
    });
  }
  const configuredVariants = functionConfig?.variants ?? undefined;
  const { map, setPromise } = useClientInferences(
    functionName,
    datapoints,
    inputs,
    variants,
    serverInferences,
    functionConfig,
  );

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

    setSearchParams(newParams, { replace: true });
  };

  return (
    <PageLayout>
      <PageHeader name="Playground" />
      <div className="flex max-w-180 flex-col gap-2">
        <Label>Function</Label>
        <FunctionSelector
          selected={functionName}
          onSelect={(value) =>
            // If the function is changed, we should reset all selectors since
            // the variant is no longer valid and the dataset may not be.
            updateSearchParams({
              functionName: value,
              variants: null,
              datasetName: null,
            })
          }
          functions={useAllFunctionConfigs()}
        />
      </div>
      <div className="flex max-w-180 flex-col gap-2">
        <Label>Dataset</Label>
        <DatasetSelector
          functionName={functionName ?? undefined}
          disabled={!functionName}
          selected={datasetName ?? undefined}
          onSelect={(value) => updateSearchParams({ datasetName: value })}
          allowCreation={false}
        />
      </div>
      <div className="flex max-w-180 flex-col gap-2">
        <Label>Variants</Label>
        <BuiltinVariantFilter
          variants={variants}
          updateSearchParams={updateSearchParams}
          builtInVariantNames={
            configuredVariants ? Object.keys(configuredVariants) : []
          }
          disabled={!functionName || !datasetName}
        />
      </div>
      {variants.length > 0 &&
        datapoints &&
        datapoints.length > 0 &&
        datasetName &&
        inputs &&
        functionName &&
        functionConfig && (
          <>
            <div className="overflow-x-auto rounded border">
              <div className="min-w-fit">
                {/* Header row with sticky positioning */}
                <div className="bg-background sticky top-0 z-20 grid grid-cols-[400px_1fr] border-b">
                  <div className="bg-background sticky left-0 z-30 flex items-center border-r p-4 font-medium">
                    Datapoints
                  </div>
                  <div className="grid auto-cols-[minmax(480px,1fr)] grid-flow-col">
                    {variants.map((variant) => {
                      const isEditable =
                        variant.type === "edited" ||
                        functionConfig?.variants?.[variant.name]?.inner.type ===
                          "chat_completion";
                      return (
                        <div
                          key={variant.name}
                          className="flex items-center gap-2 border-r p-4 font-mono font-medium last:border-r-0"
                        >
                          <div className="flex min-w-0 flex-1 items-center gap-2">
                            {variant.type === "builtin" ? (
                              <Link
                                to={`/observability/functions/${encodeURIComponent(functionName)}/variants/${encodeURIComponent(variant.name)}`}
                                className="min-w-0 truncate font-mono text-blue-600 hover:text-blue-800 hover:underline"
                                title={variant.name}
                              >
                                {getDisplayVariantName(variant)}
                              </Link>
                            ) : (
                              <span
                                className="min-w-0 truncate font-mono text-gray-500"
                                title={variant.name}
                              >
                                {getDisplayVariantName(variant)}
                              </span>
                            )}
                            <EditButton
                              onClick={() => {
                                setEditingVariant(variant);
                              }}
                              disabled={!isEditable}
                              tooltip={
                                isEditable
                                  ? "Edit variant"
                                  : "Editing is currently only supported for chat completion variants."
                              }
                            />
                          </div>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="shrink-0"
                            onClick={() => {
                              updateSearchParams({
                                variants: JSON.stringify(
                                  variants.filter((v) => v !== variant),
                                ),
                              });
                            }}
                          >
                            <span className="sr-only">
                              Remove {variant.name}
                            </span>
                            <X aria-hidden />
                          </Button>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {datapoints.map(
                  (datapoint: TensorZeroDatapoint, index: number) => (
                    <div
                      key={datapoint.id}
                      className="grid grid-cols-[400px_1fr] border-b last:border-b-0"
                    >
                      <div className="bg-background sticky left-0 z-10 flex flex-col gap-2 border-r p-4 text-sm">
                        <div className="text-xs font-medium text-gray-500">
                          Datapoint:{" "}
                          <Link
                            to={`/datasets/${encodeURIComponent(datasetName)}/datapoint/${datapoint.id}`}
                            className="font-mono text-xs text-blue-600 hover:text-blue-800 hover:underline"
                          >
                            {datapoint.id}
                          </Link>
                        </div>
                        <div>
                          <h3 className="mb-2 text-sm font-medium text-gray-500">
                            Input
                          </h3>
                          <InputSnippet
                            messages={inputs[index].messages}
                            system={inputs[index].system}
                          />
                        </div>
                        <div>
                          <h3 className="mb-2 text-sm font-medium text-gray-500">
                            Reference Output
                          </h3>
                          {datapoint.output ? (
                            <Output output={datapoint.output} />
                          ) : (
                            <div className="text-sm text-gray-500">None</div>
                          )}
                        </div>
                      </div>
                      <div className="grid auto-cols-[minmax(320px,1fr)] grid-flow-col">
                        {variants.map((variant) => {
                          return (
                            <div
                              key={`${datapoint.id}-${variant}`}
                              className="border-r p-4 last:border-r-0"
                            >
                              <DatapointPlaygroundOutput
                                datapoint={datapoint}
                                variant={variant}
                                isLoading={loadingVariants.has(variant.name)}
                                inferencePromise={map
                                  .get(variant.name)
                                  ?.get(datapoint.id)}
                                setPromise={setPromise}
                                input={inputs[index]}
                                functionName={functionName}
                                functionConfig={functionConfig}
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
          </>
        )}
      {editingVariant &&
        (() => {
          // First check if it's an edited variant
          const variantInfo = (() => {
            switch (editingVariant.type) {
              case "builtin":
                return configuredVariants?.[editingVariant.name];
              case "edited":
                return editingVariant.config;
              default:
                return undefined;
            }
          })();
          if (!variantInfo) {
            throw new Error(
              `Failed to get VariantInfo for ${editingVariant.name}`,
            );
          }
          return (
            <VariantEditor
              key={editingVariant.name}
              variantInfo={variantInfo}
              confirmVariantInfo={(newVariantInfo) => {
                const newVariantName = getNewVariantName(editingVariant.name);
                const newPlaygroundVariantInfo = {
                  type: "edited",
                  name: newVariantName,
                  config: newVariantInfo,
                };
                const newVariants = variants.map((variant) =>
                  variant.name === editingVariant.name
                    ? newPlaygroundVariantInfo
                    : variant,
                );
                setEditingVariant(null);
                updateSearchParams({
                  variants: JSON.stringify(newVariants),
                });
              }}
              isOpen={true}
              onClose={() => setEditingVariant(null)}
              variantName={
                editingVariant.name
                  ? getCleanVariantName(editingVariant)
                  : undefined
              }
            />
          );
        })()}
    </PageLayout>
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
  variants: PlaygroundVariantInfo[],
  serverInferences: NestedPromiseMap<InferenceResponse>,
  functionConfig: FunctionConfig | null,
) {
  const { map, setPromise, setMap } =
    useNestedPromiseMap<InferenceResponse>(serverInferences);

  // Single combined effect to handle both server inferences and client inferences
  useEffect(() => {
    if (!functionName || !datapoints || !inputs || !functionConfig) return;

    // First check if we need any updates
    let needsUpdate = false;
    const updates: Array<{
      variant: PlaygroundVariantInfo;
      datapoint: TensorZeroDatapoint;
      input: DisplayInput;
    }> = [];

    // Use a ref to access the current map without including it in dependencies
    setMap((prevMap) => {
      // Check each required combination
      variants.forEach((variant) => {
        const variantMap = prevMap.get(variant.name);

        datapoints.forEach((datapoint, index) => {
          const existingPromise = variantMap?.get(datapoint.id);
          if (!existingPromise) {
            needsUpdate = true;
            updates.push({
              variant,
              datapoint,
              input: inputs[index],
            });
          }
        });
      });

      // Only create a new map if we have updates
      if (!needsUpdate) {
        return prevMap; // Return the same reference to avoid re-render
      }

      const newMap = new Map(prevMap);

      // Apply updates
      updates.forEach(({ variant, datapoint, input }) => {
        let variantMap = newMap.get(variant.name);
        if (!variantMap) {
          variantMap = new Map();
          newMap.set(variant.name, variantMap);
        }
        const inferenceRequest = preparePlaygroundInferenceRequest(
          variant,
          functionName,
          datapoint,
          input,
          functionConfig,
        );
        const formData = new FormData();
        formData.append("data", JSON.stringify(inferenceRequest));
        const responsePromise = fetch("/api/tensorzero/inference", {
          method: "POST",
          body: formData,
        }).then(async (response) => {
          const data = await response.json();
          if (data.error) {
            throw new Error(data.error);
          }
          return data;
        });
        variantMap.set(datapoint.id, responsePromise);
      });

      return newMap;
    });
  }, [functionName, datapoints, inputs, variants, setMap, functionConfig]);

  return { map, setPromise, setMap };
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  if (isRouteErrorResponse(error)) {
    return (
      <PageLayout>
        <div className="flex min-h-[50vh] flex-col items-center justify-center">
          <div className="text-center">
            <h1 className="text-4xl font-bold text-gray-900">
              {error.status} {error.statusText}
            </h1>
            <p className="mt-4 text-lg text-gray-600">{error.data}</p>
            <Link
              to="/playground"
              className="mt-6 inline-block rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
            >
              Go to Playground
            </Link>
          </div>
        </div>
      </PageLayout>
    );
  } else if (error instanceof Error) {
    return (
      <PageLayout>
        <div className="flex min-h-[50vh] flex-col items-center justify-center">
          <div className="text-center">
            <h1 className="text-4xl font-bold text-gray-900">Error</h1>
            <p className="mt-4 text-lg text-gray-600">{error.message}</p>
            <details className="mt-4 max-w-2xl text-left">
              <summary className="cursor-pointer text-sm text-gray-500">
                Stack trace
              </summary>
              <pre className="mt-2 overflow-auto rounded bg-gray-100 p-4 text-xs">
                {error.stack}
              </pre>
            </details>
            <Link
              to="/playground"
              className="mt-6 inline-block rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
            >
              Go to Playground
            </Link>
          </div>
        </div>
      </PageLayout>
    );
  } else {
    return (
      <PageLayout>
        <div className="flex min-h-[50vh] flex-col items-center justify-center">
          <div className="text-center">
            <h1 className="text-4xl font-bold text-gray-900">Unknown Error</h1>
            <p className="mt-4 text-lg text-gray-600">
              An unexpected error occurred. Please try again.
            </p>
            <Link
              to="/playground"
              className="mt-6 inline-block rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
            >
              Go to Playground
            </Link>
          </div>
        </div>
      </PageLayout>
    );
  }
}
