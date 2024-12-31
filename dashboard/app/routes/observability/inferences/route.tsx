import {
  queryInferenceTable,
  queryInferenceTableBounds,
} from "~/utils/clickhouse";
import type { Route } from "./+types/route";
import InferencesTable from "./InferencesTable";
import { data, isRouteErrorResponse } from "react-router";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const before = url.searchParams.get("before");
  const after = url.searchParams.get("after");
  const pageSize = Number(url.searchParams.get("page_size")) || 10;
  if (pageSize > 100) {
    throw data("Page size cannot exceed 100", { status: 400 });
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
  console.log(loaderData);
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
export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  console.error(error);

  if (isRouteErrorResponse(error)) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">
          {error.status} {error.statusText}
        </h1>
        <p>{error.data}</p>
      </div>
    );
  } else if (error instanceof Error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">Error</h1>
        <p>{error.message}</p>
      </div>
    );
  } else {
    return (
      <div className="flex h-screen items-center justify-center text-red-500">
        <h1 className="text-2xl font-bold">Unknown Error</h1>
      </div>
    );
  }
}
