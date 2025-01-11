import {
  queryInferenceTable,
  queryInferenceTableBounds,
} from "~/utils/clickhouse/inference";
import type { Route } from "./+types/route";
import InferencesTable from "./InferencesTable";
import { data, isRouteErrorResponse } from "react-router";
import { useNavigate } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import InferenceSearchBar from "./InferenceSearchBar";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const before = url.searchParams.get("before");
  const after = url.searchParams.get("after");
  const pageSize = Number(url.searchParams.get("pageSize")) || 10;
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
  const { inferences, pageSize, bounds } = loaderData;
  const navigate = useNavigate();

  const topInference = inferences[0];
  const bottomInference = inferences[inferences.length - 1];

  const handleNextPage = () => {
    navigate(`?before=${bottomInference.id}&pageSize=${pageSize}`);
  };

  const handlePreviousPage = () => {
    navigate(`?after=${topInference.id}&pageSize=${pageSize}`);
  };

  // These are swapped because the table is sorted in descending order
  const disablePrevious = bounds.last_id === topInference.id;
  const disableNext = bounds.first_id === bottomInference.id;

  return (
    <div className="container mx-auto px-4 py-8">
      <h2 className="mb-4 text-2xl font-semibold">Inferences</h2>
      <div className="mb-6 h-px w-full bg-gray-200"></div>
      <InferenceSearchBar />
      <div className="my-6 h-px w-full bg-gray-200"></div>
      <InferencesTable inferences={inferences} />
      <PageButtons
        onPreviousPage={handlePreviousPage}
        onNextPage={handleNextPage}
        disablePrevious={disablePrevious}
        disableNext={disableNext}
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
