import { redirect, useLoaderData } from "react-router";
import type { LoaderFunctionArgs } from "react-router";
import BasicInfo from "./BasicInfo";
import { useConfig } from "~/context/config";

export async function loader({ params }: LoaderFunctionArgs) {
  const { function_name, variant_name } = params;
  if (!function_name || !variant_name) {
    return redirect("/observability/functions");
  }
  return { function_name, variant_name };
}

export default function VariantDetails() {
  const { function_name, variant_name } = useLoaderData<typeof loader>();
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
  return (
    <div className="container mx-auto px-4 py-8">
      <h2 className="mb-4 text-2xl font-semibold">
        Function{" "}
        <code className="rounded bg-gray-100 p-1 text-2xl">
          {function_name}
        </code>
      </h2>
      <div className="mb-6 h-px w-full bg-gray-200"></div>
      <BasicInfo variantConfig={variant_config} />
      <div className="mb-6 h-px w-full bg-gray-200"></div>
    </div>
  );
}
