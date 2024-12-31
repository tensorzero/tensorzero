import {
  queryInferenceTable,
  queryInferenceTableBounds,
} from "~/utils/clickhouse";
import type { Route } from "./+types/route";
import InferencesTable from "./InferencesTable";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const before = url.searchParams.get("before");
  const after = url.searchParams.get("after");
  const pageSize = Number(url.searchParams.get("page_size")) || 10;
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
