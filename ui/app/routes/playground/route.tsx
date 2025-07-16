import { useSearchParams, useNavigate, data } from "react-router";
import { DatasetSelector } from "~/components/dataset/DatasetSelector";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { PageHeader, PageLayout } from "~/components/layout/PageLayout";
import { useConfig } from "~/context/config";
import { getConfig } from "~/utils/config/index.server";
import type { Route } from "./+types/route";
import { listDatapoints } from "~/utils/tensorzero.server";
import { VariantFilter } from "~/components/function/variant/variant-filter";
import {
  prepareInferenceActionRequest,
  tensorZeroResolvedInputToInput,
  useInferenceActionFetcher,
} from "~/routes/api/tensorzero/inference.utils";
import { resolveInput } from "~/utils/resolve.server";
import { Loader2 } from "lucide-react";
import type { Datapoint as TensorZeroDatapoint } from "tensorzero-node";
import type { DisplayInput } from "~/utils/clickhouse/common";
import { useEffect } from "react";

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
  const config = await getConfig();
  const functionConfig = functionName ? config.functions[functionName] : null;
  if (functionName && !functionConfig) {
    throw data(`Function config not found for function ${functionName}`, {
      status: 404,
    });
  }
  const datasetName = searchParams.get("datasetName");
  const datapoints = datasetName
    ? await listDatapoints(
        datasetName,
        functionName ?? undefined,
        limit,
        offset,
      )
    : undefined;
  const inputs = datapoints
    ? await Promise.all(
        datapoints.map(async (datapoint) => {
          const inputData = tensorZeroResolvedInputToInput(datapoint.input);
          return await resolveInput(inputData, functionConfig ?? null);
        }),
      )
    : undefined;
  return { functionName, datasetName, datapoints, inputs };
}

export default function PlaygroundPage({ loaderData }: Route.ComponentProps) {
  const { functionName, datasetName, datapoints, inputs } = loaderData;
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
        onSelect={(value) => updateSearchParams({ functionName: value })}
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
                      {selectedVariants.map((variant) => (
                        <td
                          key={`${datapoint.id}-${variant}`}
                          className={`p-4 ${
                            selectedVariants.length <= 3 ? "" : "w-80 min-w-80"
                          }`}
                        >
                          <DatapointPlaygroundOutput
                            datapoint={datapoint}
                            input={inputs[index]}
                            functionName={functionName}
                            variantName={variant}
                          />
                        </td>
                      ))}
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
}
function DatapointPlaygroundOutput({
  datapoint,
  input,
  functionName,
  variantName,
}: DatapointPlaygroundOutputProps) {
  const variantInferenceFetcher = useInferenceActionFetcher();
  const variantSource = "clickhouse_datapoint";
  const variantInferenceIsLoading =
    variantInferenceFetcher.state === "submitting" ||
    variantInferenceFetcher.state === "loading";

  const { submit, data, state } = variantInferenceFetcher;
  useEffect(() => {
    const request = prepareInferenceActionRequest({
      source: variantSource,
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
    const formData = new FormData();
    formData.append("data", JSON.stringify(request));
    submit(formData);
  }, [data, state, variantSource, input, functionName, variantName, datapoint]);
  console.log(variantInferenceFetcher.data);

  return (
    <>
      {variantInferenceIsLoading ? (
        <div className="flex h-32 items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin" />
        </div>
      ) : variantInferenceFetcher.error ? (
        <div className="flex h-32 items-center justify-center">
          <div className="text-center text-red-600">
            <p className="font-semibold">Error</p>
            <p className="text-sm">{variantInferenceFetcher.error?.message}</p>
          </div>
        </div>
      ) : (
        <div className="flex h-32 items-center justify-center">
          <div className="text-muted-foreground text-sm">
            Output will appear here.
          </div>
        </div>
      )}
    </>
  );
}
