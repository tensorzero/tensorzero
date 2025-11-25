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
import {
  useFunctionConfig,
  useAllFunctionConfigs,
  useConfig,
} from "~/context/config";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";
import type { Route } from "./+types/route";
import { listDatapoints } from "~/utils/tensorzero.server";
import { datapointInputToZodInput } from "~/routes/api/tensorzero/inference.utils";
import { resolveInput } from "~/utils/resolve.server";
import { X } from "lucide-react";
import type { Datapoint as TensorZeroDatapoint } from "~/types/tensorzero";
import { useMemo, useState } from "react";
import { Button } from "~/components/ui/button";
import PageButtons from "~/components/utils/PageButtons";
import { countDatapointsForDatasetFunction } from "~/utils/clickhouse/datasets.server";
import Input from "~/components/inference/Input";
import { Output } from "~/components/inference/Output";
import { Label } from "~/components/ui/label";
import DatapointPlaygroundOutput from "./DatapointPlaygroundOutput";
import { safeParseInt, symmetricDifference } from "~/utils/common";
import { EditButton } from "~/components/utils/EditButton";
import { VariantEditor } from "~/components/function/variant/VariantEditor";
import { Badge } from "~/components/ui/badge";
import {
  extractOriginalVariantNameFromEdited,
  getClientInferenceQueryFunction,
  getClientInferenceQueryKey,
  getNewVariantName,
  getVariants,
  type ClientInferenceInputArgs,
  type PlaygroundVariantInfo,
} from "./utils";
import { BuiltinVariantFilter } from "./BuiltInVariantSelector";
import {
  dehydrate,
  HydrationBoundary,
  QueryClient,
} from "@tanstack/react-query";
import { toDatapointUrl } from "~/utils/urls";
import clsx from "clsx";

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
  throw new Error(`Unknown variant type: ${JSON.stringify(variant)}`);
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
  throw new Error(`Unknown variant type: ${JSON.stringify(variant)}`);
}

export function shouldRevalidate(arg: ShouldRevalidateFunctionArgs) {
  const { currentUrl, nextUrl } = arg;
  // First check that the base route is the same
  if (currentUrl.pathname !== nextUrl.pathname) {
    return true;
  }

  // Copy search params for comparison
  const currentSearchParams = new URLSearchParams(currentUrl.searchParams);
  const nextSearchParams = new URLSearchParams(nextUrl.searchParams);

  // Remove variants from comparison since we want to skip revalidation if only
  // the variants changed
  currentSearchParams.delete("variants");
  nextSearchParams.delete("variants");

  // Then manually compare the remaining search params and revalidate if
  // anything else changed
  if (currentSearchParams.size !== nextSearchParams.size) {
    // number of search params has changed; revalidate
    return true;
  }

  const currentKeys = new Set(currentSearchParams.keys());
  const nextKeys = new Set(nextSearchParams.keys());
  if (symmetricDifference(currentKeys, nextKeys).size > 0) {
    // search param keys have changed; revalidate
    return true;
  }

  for (const [key, value] of currentSearchParams.entries()) {
    const nextValue = nextSearchParams.get(key);
    if (nextValue !== value) {
      // search param values have changed; revalidate
      return true;
    }
  }

  // No other changes detected; skip revalidation
  return false;
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
            const inputData = datapointInputToZodInput(datapoint.input);
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

  const queryClient = new QueryClient();
  const variants = getVariants(searchParams);
  if (
    variants.length > 0 &&
    datapoints &&
    datapoints.length > 0 &&
    datasetName &&
    inputs &&
    functionName &&
    functionConfig
  ) {
    await Promise.all(
      datapoints.flatMap((datapoint, index) =>
        variants.map(async (variant) => {
          const input = inputs[index];
          const args: ClientInferenceInputArgs = {
            datapoint,
            functionConfig,
            functionName,
            input,
            variant,
            toolsConfig: config.tools,
          };
          return queryClient.prefetchQuery({
            queryKey: getClientInferenceQueryKey(args),
            queryFn: getClientInferenceQueryFunction(args),
          });
        }),
      ),
    );
  }

  return {
    functionName,
    datasetName,
    datapoints,
    inputs,
    totalDatapoints,
    offset,
    limit,
    dehydratedState: dehydrate(queryClient),
  };
}

export default function PlaygroundPage({ loaderData }: Route.ComponentProps) {
  const navigation = useNavigation();
  const [currentSearchParams, setSearchParams] = useSearchParams();
  const [editingVariant, setEditingVariant] =
    useState<PlaygroundVariantInfo | null>(null);
  const { variants, searchParams } = useMemo(() => {
    if (navigation.state !== "loading") {
      return {
        variants: getVariants(currentSearchParams),
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
      variants: getVariants(nextSearchParams),
      searchParams: nextSearchParams,
      loadingVariants,
    };
  }, [navigation, currentSearchParams]);

  const {
    functionName,
    datasetName,
    datapoints,
    inputs,
    totalDatapoints,
    offset,
    limit,
    dehydratedState,
  } = loaderData;
  const functionConfig = useFunctionConfig(functionName);
  const config = useConfig();
  if (functionName && !functionConfig) {
    throw data(`Function config not found for function ${functionName}`, {
      status: 404,
    });
  }
  const configuredVariants = functionConfig?.variants ?? undefined;

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
      {datapoints &&
        datapoints.length > 0 &&
        datasetName &&
        inputs &&
        functionName &&
        functionConfig && (
          <>
            <div className="overflow-x-auto rounded border">
              <div className="min-w-fit">
                {/* Header row with sticky positioning */}
                <GridRow as="header" variantCount={variants.length}>
                  <div className="bg-background sticky left-0 z-30 flex items-center border-r p-4 font-medium">
                    Datapoints
                  </div>
                  {variants.length > 0 && (
                    <div className="grid auto-cols-[minmax(480px,1fr)] grid-flow-col">
                      {variants.map((variant) => {
                        const isEditable =
                          variant.type === "edited" ||
                          functionConfig?.variants?.[variant.name]?.inner
                            .type === "chat_completion";
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
                  )}
                </GridRow>

                <HydrationBoundary state={dehydratedState}>
                  {datapoints.map(
                    (datapoint: TensorZeroDatapoint, index: number) => (
                      <GridRow
                        key={datapoint.id}
                        variantCount={variants.length}
                      >
                        <div className="bg-background sticky left-0 z-10 flex flex-col gap-2 border-r p-4 text-sm">
                          <div className="text-xs font-medium text-gray-500">
                            Datapoint:{" "}
                            <Link
                              to={toDatapointUrl(datasetName, datapoint.id)}
                              className="font-mono text-xs text-blue-600 hover:text-blue-800 hover:underline"
                            >
                              {datapoint.id}
                            </Link>
                          </div>
                          <div>
                            <h3 className="mb-2 text-sm font-medium text-gray-500">
                              Input
                            </h3>
                            <Input
                              messages={inputs[index].messages}
                              system={inputs[index].system}
                              maxHeight={150}
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
                        {variants.length > 0 && (
                          <div className="grid auto-cols-[minmax(320px,1fr)] grid-flow-col">
                            {variants.map((variant) => {
                              return (
                                <div
                                  key={`${datapoint.id}-${variant.name}`}
                                  className="border-r p-4 last:border-r-0"
                                >
                                  <DatapointPlaygroundOutput
                                    datapoint={datapoint}
                                    variant={variant}
                                    input={inputs[index]}
                                    functionName={functionName}
                                    functionConfig={functionConfig}
                                    toolsConfig={config.tools}
                                  />
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </GridRow>
                    ),
                  )}
                </HydrationBoundary>
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

function GridRow({
  as = "datapoint",
  variantCount,
  ...props
}: {
  as?: "header" | "datapoint";
  variantCount: number;
  children: React.ReactNode;
}) {
  return (
    <div
      {...props}
      className={clsx(
        "grid border-b",
        variantCount > 0 && "grid-cols-[400px_1fr]",
        as === "header" && "bg-background sticky top-0 z-20",
        as === "datapoint" && "last:border-b-0",
      )}
    />
  );
}
