import { useSearchParams, useNavigate, data } from "react-router";
import { DatasetSelector } from "~/components/dataset/DatasetSelector";
import { FunctionSelector } from "~/components/function/FunctionSelector";
import { PageHeader, PageLayout } from "~/components/layout/PageLayout";
import { useConfig } from "~/context/config";
import { getConfig } from "~/utils/config/index.server";
import type { Route } from "./+types/route";
import { listDatapoints } from "~/utils/tensorzero.server";
import { VariantFilter } from "~/components/function/variant/variant-filter";

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
  if (functionName && !config.functions[functionName]) {
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
  return { functionName, datasetName, datapoints };
}

export default function PlaygroundPage({ loaderData }: Route.ComponentProps) {
  const { functionName, datasetName, datapoints } = loaderData;
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
      <div>
        {datapoints?.map((datapoint) => (
          <div key={datapoint.id}>{datapoint.id}</div>
        ))}
      </div>
    </PageLayout>
  );
}
