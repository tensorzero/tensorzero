import {
  queryInferenceTable,
  queryInferenceTableBounds,
  countInferencesByFunction,
} from "~/utils/clickhouse/inference.server";
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
import { logger } from "~/utils/logger";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const before = url.searchParams.get("before");
  const after = url.searchParams.get("after");
  const limit = Number(url.searchParams.get("limit")) || 10;
  if (limit > 100) {
    throw data("Limit cannot exceed 100", { status: 400 });
  }

  const [inferences, bounds, countsInfo] = await Promise.all([
    queryInferenceTable({
      before: before || undefined,
      after: after || undefined,
      limit,
    }),
    queryInferenceTableBounds(),
    countInferencesByFunction(),
  ]);

  const totalInferences = countsInfo.reduce((acc, curr) => acc + curr.count, 0);

  return {
    inferences,
    limit,
    bounds,
    totalInferences,
  };
}

export default function InferencesPage({ loaderData }: Route.ComponentProps) {
  const { inferences, limit, bounds, totalInferences } = loaderData;

  const navigate = useNavigate();

  const topInference = inferences.at(0);
  const bottomInference = inferences.at(inferences.length - 1);

  const handleNextPage = () => {
    if (bottomInference) {
      navigate(`?before=${bottomInference.id}&limit=${limit}`, {
        preventScrollReset: true,
      });
    }
  };

  const handlePreviousPage = () => {
    if (topInference) {
      navigate(`?after=${topInference.id}&limit=${limit}`, {
        preventScrollReset: true,
      });
    }
  };

  // These are swapped because the table is sorted in descending order
  const disablePrevious =
    !bounds?.last_id || bounds.last_id === topInference?.id;
  const disableNext =
    !bounds?.first_id || bounds.first_id === bottomInference?.id;

  return (
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
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  logger.error(error);

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
