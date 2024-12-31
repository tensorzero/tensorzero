import {
  queryInferenceTable,
  queryInferenceTableBounds,
} from "~/utils/clickhouse";
import type { Route } from "./+types/route";
import InferencesTable from "./InferencesTable";
import { data } from "react-router";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const before = url.searchParams.get("before");
  const after = url.searchParams.get("after");
  const pageSize = Number(url.searchParams.get("page_size")) || 10;
  if (pageSize > 100) {
    return data("Page size cannot exceed 100", { status: 400 });
  }

  const [inferences, bounds] = await Promise.all([
    queryInferenceTable({
      before: before || undefined,
      after: after || undefined,
      page_size: pageSize,
    }),
    queryInferenceTableBounds(),
  ]);

  return {
    inferences,
    pageSize,
    bounds,
  };
}

export default function InferencesPage({ loaderData }: Route.ComponentProps) {
  if (typeof loaderData === "string") {
    return (
      <div className="flex h-screen items-center justify-center text-red-500">
        {loaderData}
      </div>
    );
  }

  const { inferences, pageSize, bounds } = loaderData;

  return (
    <div className="container mx-auto px-4 py-8">
      <InferencesTable
        inferences={inferences}
        pageSize={pageSize}
        bounds={bounds}
      />
    </div>
  );
}
