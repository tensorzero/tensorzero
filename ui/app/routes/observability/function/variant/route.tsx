import { useLoaderData } from "react-router";
import type { LoaderFunctionArgs } from "react-router";

export async function loader({ params }: LoaderFunctionArgs) {
  const { function_name, variant_name } = params;
  return { function_name, variant_name };
}

export default function VariantDetails() {
  const { function_name, variant_name } = useLoaderData<typeof loader>();
  return (
    <div>
      Variant Details for function <code>{function_name}</code> variant{" "}
      <code>{variant_name}</code>
    </div>
  );
}
