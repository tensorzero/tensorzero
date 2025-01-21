import { data, redirect, useLoaderData, useNavigate } from "react-router";
import type { LoaderFunctionArgs } from "react-router";
import BasicInfo from "./BasicInfo";
import { useConfig } from "~/context/config";
import PageButtons from "~/components/utils/PageButtons";
import { Badge } from "~/components/ui/badge";
import VariantInferenceTable from "./VariantInferenceTable";
import { getConfig } from "~/utils/config/index.server";
import {
  countInferencesForVariant,
  queryInferenceTableBoundsByVariantName,
  queryInferenceTableByVariantName,
} from "~/utils/clickhouse/inference";

export async function loader({ request, params }: LoaderFunctionArgs) {
  const { function_name, variant_name } = params;
  if (!function_name || !variant_name) {
    return redirect("/observability/functions");
  }
  const config = await getConfig();
  const url = new URL(request.url);
  const beforeInference = url.searchParams.get("beforeInference");
  const afterInference = url.searchParams.get("afterInference");
  const pageSize = Number(url.searchParams.get("pageSize")) || 10;
  if (pageSize > 100) {
    throw data("Page size cannot exceed 100", { status: 400 });
  }
  const [num_inferences, inferences, inference_bounds] = await Promise.all([
    countInferencesForVariant(
      function_name,
      config.functions[function_name],
      variant_name,
    ),
    queryInferenceTableByVariantName({
      function_name,
      variant_name,
      page_size: pageSize,
      before: beforeInference || undefined,
      after: afterInference || undefined,
    }),
    queryInferenceTableBoundsByVariantName({
      function_name,
      variant_name,
    }),
  ]);
  return {
    function_name,
    variant_name,
    num_inferences,
    inferences,
    inference_bounds,
  };
}

export default function VariantDetails() {
  const {
    function_name,
    variant_name,
    num_inferences,
    inferences,
    inference_bounds,
  } = useLoaderData<typeof loader>();
  const navigate = useNavigate();
  const config = useConfig();
  const function_config = config.functions[function_name];
  if (!function_config) {
    throw new Response(
      "Function not found. This likely means there is data in ClickHouse from an old TensorZero config.",
      {
        status: 404,
        statusText: "Not Found",
      },
    );
  }
  const variant_config = function_config.variants[variant_name];
  if (!variant_config) {
    throw new Response(
      "Variant not found. This likely means there is data in ClickHouse from an old TensorZero config.",
      {
        status: 404,
        statusText: "Not Found",
      },
    );
  }

  const topInference = inferences[0];
  const bottomInference = inferences[inferences.length - 1];
  const handleNextInferencePage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("afterInference");
    searchParams.set("beforeInference", bottomInference.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousInferencePage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("beforeInference");
    searchParams.set("afterInference", topInference.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };
  // These are swapped because the table is sorted in descending order
  const disablePreviousInferencePage =
    inference_bounds.last_id === topInference.id;
  const disableNextInferencePage =
    inference_bounds.first_id === bottomInference.id;
  return (
    <div className="container mx-auto px-4 py-8">
      <h2 className="mb-4 text-2xl font-semibold">
        Variant{" "}
        <code className="rounded bg-gray-100 p-1 text-2xl">{variant_name}</code>
      </h2>
      <div className="mb-6 h-px w-full bg-gray-200"></div>
      <BasicInfo variantConfig={variant_config} function_name={function_name} />
      <div className="mb-6 h-px w-full bg-gray-200"></div>
      <div className="mt-6">
        <h3 className="mb-2 flex items-center gap-2 text-xl font-semibold">
          Inferences
          <Badge variant="secondary">Count: {num_inferences}</Badge>
        </h3>
        <VariantInferenceTable inferences={inferences} />
        <PageButtons
          onPreviousPage={handlePreviousInferencePage}
          onNextPage={handleNextInferencePage}
          disablePrevious={disablePreviousInferencePage}
          disableNext={disableNextInferencePage}
        />
      </div>
    </div>
  );
}
