import {
  useSearchParams,
  useNavigate,
  data,
  Await,
  useFetcher,
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

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const functionName = searchParams.get("functionName");
  const limit = searchParams.get("limit")
    ? parseInt(searchParams.get("limit")!)
    : 10;
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
  const datapoints = datasetName
    ? await listDatapoints(
        datasetName,
        functionName ?? undefined,
        limit,
        offset,
      )
    : undefined;
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
  const {
    type,
    functionName,
    datasetName,
    datapoints,
    inputs,
    serverInferences,
  } = loaderData;
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const config = useConfig();

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
    : undefined;

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
      <FunctionSelector
        selected={functionName}
        onSelect={(value) =>
          updateSearchParams({ functionName: value, variant: null })
        }
        functions={config.functions}
      />
      <DatasetSelector
        selected={datasetName ?? undefined}
        onSelect={(value) => updateSearchParams({ datasetName: value })}
        allowCreation={false}
      />
      {functionName && datasetName && variantData && (
        <VariantFilter
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
      )}
      {selectedVariants.length > 0 &&
        datapoints &&
        datapoints.length > 0 &&
        inputs &&
        functionName && (
          <div className="-mx-8 mt-6 md:-mx-16 lg:-mx-32 xl:-mx-48 2xl:-mx-64">
            <div className="mx-auto h-[calc(100vh-200px)] overflow-auto px-4">
              <table
                className={selectedVariants.length <= 3 ? "w-full" : ""}
                style={{ borderCollapse: "collapse" }}
              >
                <thead className="sticky top-0 z-20">
                  <tr>
                    <th className="bg-background sticky left-0 z-30 border-r border-b p-4 text-left font-medium">
                      Datapoint ID
                    </th>
                    {selectedVariants.map((variant) => (
                      <th
                        key={variant}
                        className={`bg-background border-b p-4 text-left font-medium ${
                          selectedVariants.length <= 3 ? "" : "w-80 min-w-80"
                        }`}
                      >
                        {variant}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {datapoints.map((datapoint, index) => (
                    <tr key={datapoint.id} className="border-t">
                      <td className="bg-background sticky left-0 z-10 border-r p-4 font-mono text-sm">
                        {datapoint.id}
                      </td>
                      {selectedVariants.map((variant, variantIndex) => {
                        const inferenceIndex =
                          index * selectedVariants.length + variantIndex;
                        return (
                          <td
                            key={`${datapoint.id}-${variant}`}
                            className={`p-4 ${
                              selectedVariants.length <= 3
                                ? ""
                                : "w-80 min-w-80"
                            }`}
                          >
                            <DatapointPlaygroundOutput
                              datapoint={datapoint}
                              input={inputs[index]}
                              functionName={functionName}
                              variantName={variant}
                              serverInference={
                                serverInferences?.[inferenceIndex]
                              }
                            />
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
    </PageLayout>
  );
}

interface DatapointPlaygroundOutputProps {
  datapoint: TensorZeroDatapoint;
  input: DisplayInput;
  functionName: string;
  variantName: string;
  serverInference: Promise<InferenceResponse> | undefined;
}
function DatapointPlaygroundOutput({
  datapoint,
  input,
  functionName,
  variantName,
  serverInference,
}: DatapointPlaygroundOutputProps) {
  const fetcher = useFetcher();
  if (!serverInference) {
    return (
      <div className="flex h-32 items-center justify-center">
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
          const url = new URL(window.location.href);
          const queryParams = new URLSearchParams(url.search);
          // Set the refresh params
          queryParams.set("refreshDatapointId", datapoint.id);
          queryParams.set("refreshVariantName", variantName);
          url.search = queryParams.toString();
          const result = fetcher.load(url.toString());
        }}
      >
        <Refresh />
      </Button>

      <Suspense
        fallback={
          <div className="flex h-32 items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin" />
          </div>
        }
      >
        <Await
          resolve={serverInference}
          errorElement={
            <div className="flex h-32 items-center justify-center">
              <div className="text-center text-red-600">
                <p className="font-semibold">Error</p>
                <p className="text-sm">Failed to load inference</p>
              </div>
            </div>
          }
        >
          {(response) => {
            let output;
            if ("content" in response) {
              output = response.content;
            } else {
              output = response.output;
            }
            return (
              <div className="flex h-32 items-center justify-center">
                <NewOutput output={output} />
              </div>
            );
          }}
        </Await>
      </Suspense>
    </div>
  );
}
