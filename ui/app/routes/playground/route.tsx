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
import { tensorZeroStoredInputToInput } from "~/routes/api/tensorzero/inference.utils";
import { resolveInput } from "~/utils/resolve.server";
import { X } from "lucide-react";
import type { Datapoint as TensorZeroDatapoint } from "tensorzero-node";
import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";
import { Button } from "~/components/ui/button";
import PageButtons from "~/components/utils/PageButtons";
import { countDatapointsForDatasetFunction } from "~/utils/clickhouse/datasets.server";
import InputSnippet from "~/components/inference/InputSnippet";
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
  HydrationBoundary,
  QueryClient,
  dehydrate,
  useQueryClient,
} from "@tanstack/react-query";
import type { DisplayInput } from "~/utils/clickhouse/common";
import type { FunctionConfig } from "tensorzero-node";

const DEFAULT_LIMIT = 5;
const DEFAULT_VIRTUALIZED_ROW_HEIGHT = 520;
const VIRTUALIZED_OVERSCAN = 3;
const PREFETCH_BUFFER = 5;
const SERVER_PREFETCH_LIMIT = 6;

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
            const inputData = tensorZeroStoredInputToInput(datapoint.input);
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
    const prefetchCount = Math.min(datapoints.length, SERVER_PREFETCH_LIMIT);
    const datapointsToPrefetch = datapoints.slice(0, prefetchCount);
    await Promise.all(
      datapointsToPrefetch.flatMap((datapoint, index) =>
        variants.map(async (variant) => {
          const input = inputs[index];
          const args: ClientInferenceInputArgs = {
            datapoint,
            functionConfig,
            functionName,
            input,
            variant,
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

                <HydrationBoundary state={dehydratedState}>
                  <VirtualizedDatapointList
                    datapoints={datapoints as TensorZeroDatapoint[]}
                    datasetName={datasetName as string}
                    inputs={inputs as DisplayInput[]}
                    variants={variants}
                    functionConfig={functionConfig as FunctionConfig}
                    functionName={functionName}
                  />
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

type VirtualizedDatapointListProps = {
  datapoints: TensorZeroDatapoint[];
  datasetName: string;
  inputs: DisplayInput[];
  variants: PlaygroundVariantInfo[];
  functionName: string;
  functionConfig: FunctionConfig;
};

function VirtualizedDatapointList(props: VirtualizedDatapointListProps) {
  const {
    datapoints,
    datasetName,
    inputs,
    variants,
    functionName,
    functionConfig,
  } = props;
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);
  const rowElementsRef = useRef<Map<number, HTMLDivElement>>(new Map());
  const rowHeightsRef = useRef<number[]>([]);
  const totalHeightRef = useRef(0);
  const [scrollOffset, setScrollOffset] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(0);
  const [measurementVersion, forceMeasurementUpdate] = useReducer(
    (value: number) => value + 1,
    0,
  );
  const queryClient = useQueryClient();

  useEffect(() => {
    const count = datapoints.length;
    rowHeightsRef.current = Array.from(
      { length: count },
      () => DEFAULT_VIRTUALIZED_ROW_HEIGHT,
    );
    totalHeightRef.current = count * DEFAULT_VIRTUALIZED_ROW_HEIGHT;
    resizeObserverRef.current?.disconnect();
    rowElementsRef.current.clear();
    forceMeasurementUpdate();
  }, [datapoints.length]);

  const updateRowHeight = useCallback((index: number, height: number) => {
    const heights = rowHeightsRef.current;
    const previousHeight = heights[index] ?? DEFAULT_VIRTUALIZED_ROW_HEIGHT;
    if (Math.abs(previousHeight - height) <= 1) {
      return;
    }
    heights[index] = height;
    totalHeightRef.current += height - previousHeight;
    forceMeasurementUpdate();
  }, []);

  useEffect(() => {
    if (typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const target = entry.target as HTMLElement;
        const indexAttribute = target.dataset.index;
        if (indexAttribute) {
          updateRowHeight(Number(indexAttribute), entry.contentRect.height);
        }
      }
    });
    resizeObserverRef.current = observer;
    rowElementsRef.current.forEach((element) => observer.observe(element));
    return () => {
      observer.disconnect();
      resizeObserverRef.current = null;
    };
  }, [updateRowHeight]);

  const registerRow = useCallback(
    (index: number) => (node: HTMLDivElement | null) => {
      const elements = rowElementsRef.current;
      const observer = resizeObserverRef.current;
      const existing = elements.get(index);
      if (existing && observer) {
        observer.unobserve(existing);
      }

      if (!node) {
        elements.delete(index);
        return;
      }

      elements.set(index, node);
      node.dataset.index = `${index}`;
      observer?.observe(node);
      updateRowHeight(index, node.getBoundingClientRect().height);
    },
    [updateRowHeight],
  );

  useEffect(() => {
    const element = scrollContainerRef.current;
    if (!element) {
      return;
    }
    setViewportHeight(element.clientHeight);
    if (typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setViewportHeight(entry.contentRect.height);
      }
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
    };
  }, []);

  const variantKey = useMemo(
    () => variants.map((variant) => variant.name).join("|"),
    [variants],
  );

  useEffect(() => {
    const element = scrollContainerRef.current;
    if (element) {
      element.scrollTo({ top: 0 });
    }
    setScrollOffset(0);
  }, [datasetName, functionName, variantKey]);

  const virtualization = useMemo(() => {
    const count = datapoints.length;
    // Trigger recalculation when row measurements change even though the
    // value itself is not used directly.
    void measurementVersion;
    if (count === 0) {
      return {
        startIndex: 0,
        endIndex: 0,
        topPadding: 0,
        bottomPadding: 0,
      };
    }

    const heights = rowHeightsRef.current;
    const defaultHeight = DEFAULT_VIRTUALIZED_ROW_HEIGHT;
    let startIndex = 0;
    let accumulatedHeight = 0;
    for (let index = 0; index < count; index += 1) {
      const nextHeight = heights[index] ?? defaultHeight;
      if (accumulatedHeight + nextHeight > scrollOffset) {
        break;
      }
      accumulatedHeight += nextHeight;
      startIndex = index + 1;
    }

    let endIndex = startIndex;
    let coveredHeight = accumulatedHeight;
    const maxOffset = scrollOffset + viewportHeight;
    while (endIndex < count && coveredHeight < maxOffset) {
      const nextHeight = heights[endIndex] ?? defaultHeight;
      coveredHeight += nextHeight;
      endIndex += 1;
    }

    const renderStart = Math.max(0, startIndex - VIRTUALIZED_OVERSCAN);
    const renderEnd = Math.min(count, endIndex + VIRTUALIZED_OVERSCAN);

    let topPadding = 0;
    for (let index = 0; index < renderStart; index += 1) {
      topPadding += heights[index] ?? defaultHeight;
    }

    let renderedHeight = 0;
    for (let index = renderStart; index < renderEnd; index += 1) {
      renderedHeight += heights[index] ?? defaultHeight;
    }

    const bottomPadding = Math.max(
      totalHeightRef.current - topPadding - renderedHeight,
      0,
    );

    return {
      startIndex: renderStart,
      endIndex: renderEnd,
      topPadding,
      bottomPadding,
    };
  }, [datapoints.length, measurementVersion, scrollOffset, viewportHeight]);

  const { startIndex, endIndex, topPadding, bottomPadding } = virtualization;

  const prefetchIndices = useMemo(() => {
    const indices = new Set<number>();
    for (let index = startIndex; index < endIndex; index += 1) {
      indices.add(index);
    }
    for (let buffer = 1; buffer <= PREFETCH_BUFFER; buffer += 1) {
      const before = startIndex - buffer;
      const after = endIndex - 1 + buffer;
      if (before >= 0) {
        indices.add(before);
      }
      if (after < datapoints.length) {
        indices.add(after);
      }
    }
    return Array.from(indices).sort((a, b) => a - b);
  }, [datapoints.length, endIndex, startIndex]);

  useEffect(() => {
    prefetchIndices.forEach((index) => {
      const datapoint = datapoints[index];
      const input = inputs[index];
      if (!datapoint || !input) {
        return;
      }
      variants.forEach((variant) => {
        const args: ClientInferenceInputArgs = {
          datapoint,
          functionConfig,
          functionName,
          input,
          variant,
        };
        queryClient.prefetchQuery({
          queryKey: getClientInferenceQueryKey(args),
          queryFn: getClientInferenceQueryFunction(args),
        });
      });
    });
  }, [
    datapoints,
    functionConfig,
    functionName,
    inputs,
    prefetchIndices,
    queryClient,
    variants,
  ]);

  if (datapoints.length === 0) {
    return null;
  }

  return (
    <div
      ref={scrollContainerRef}
      className="max-h-[70vh] overflow-y-auto"
      onScroll={(event) => {
        setScrollOffset(event.currentTarget.scrollTop);
      }}
    >
      <div style={{ paddingTop: topPadding, paddingBottom: bottomPadding }}>
        {Array.from({ length: endIndex - startIndex }, (_, offset) => {
          const index = startIndex + offset;
          const datapoint = datapoints[index];
          const input = inputs[index];
          const isLastRow = index === datapoints.length - 1;
          if (!datapoint || !input) {
            return null;
          }

          return (
            <div
              key={datapoint.id}
              ref={registerRow(index)}
              className={
                "grid grid-cols-[400px_1fr] border-b" +
                (isLastRow ? " border-b-0" : "")
              }
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
                    messages={input.messages}
                    system={input.system}
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
              <div className="grid auto-cols-[minmax(320px,1fr)] grid-flow-col">
                {variants.map((variant) => (
                  <div
                    key={`${datapoint.id}-${variant.name}`}
                    className="border-r p-4 last:border-r-0"
                  >
                    <DatapointPlaygroundOutput
                      datapoint={datapoint}
                      variant={variant}
                      input={input}
                      functionName={functionName}
                      functionConfig={functionConfig}
                    />
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
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
