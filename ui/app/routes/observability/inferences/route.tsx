import {
  queryInferenceTable,
  queryInferenceTableBounds,
  countInferencesByFunction,
} from "~/utils/clickhouse/inference";
import type { Route } from "./+types/route";
import InferencesTable from "./InferencesTable";
import { data, isRouteErrorResponse, useNavigate } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import InferenceSearchBar from "./InferenceSearchBar";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const before = url.searchParams.get("before");
  const after = url.searchParams.get("after");
  const pageSize = Number(url.searchParams.get("pageSize")) || 10;
  if (pageSize > 100) {
    throw data("Page size cannot exceed 100", { status: 400 });
  }

  const [inferences, bounds, countsInfo] = await Promise.all([
    queryInferenceTable({
      before: before || undefined,
      after: after || undefined,
      page_size: pageSize,
    }),
    queryInferenceTableBounds(),
    countInferencesByFunction(),
  ]);

  const totalInferences = countsInfo.reduce((acc, curr) => acc + curr.count, 0);

  return {
    inferences,
    pageSize,
    bounds,
    totalInferences,
  };
}

export default function InferencesPage({ loaderData }: Route.ComponentProps) {
  const { inferences, pageSize, bounds, totalInferences } = loaderData;
  const navigate = useNavigate();

  if (inferences.length === 0) {
    return (
      <div className="container mx-auto px-4 pb-8">
        <PageLayout>
          <PageHeader heading="Inferences" count={totalInferences} />
          <SectionLayout>
            <InferenceSearchBar />
            <div className="py-8 text-center text-gray-500">
              No inferences found
            </div>
          </SectionLayout>
        </PageLayout>
      </div>
    );
  }

  const topInference = inferences[0];
  const bottomInference = inferences[inferences.length - 1];

  const handleNextPage = () => {
    navigate(`?before=${bottomInference.id}&pageSize=${pageSize}`, {
      preventScrollReset: true,
    });
  };

  const handlePreviousPage = () => {
    navigate(`?after=${topInference.id}&pageSize=${pageSize}`, {
      preventScrollReset: true,
    });
  };

  const disablePrevious =
    !bounds?.last_id || bounds.last_id === topInference.id;
  const disableNext =
    !bounds?.first_id || bounds.first_id === bottomInference.id;

  return (
    <div className="container mx-auto px-4 pb-8">
      <PageLayout>
        <PageHeader heading="Inferences" count={totalInferences} />
        <SectionLayout>
          <InferenceSearchBar />
          <InferencesTable inferences={inferences} />
          <PageButtons
            onPreviousPage={handlePreviousPage}
            onNextPage={handleNextPage}
            disablePrevious={disablePrevious}
            disableNext={disableNext}
          />
        </SectionLayout>
      </PageLayout>
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
