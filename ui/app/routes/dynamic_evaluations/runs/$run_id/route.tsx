import type { Route } from "./+types/route";
import {
  isRouteErrorResponse,
  useNavigate,
  type RouteHandle,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
} from "~/components/layout/PageLayout";
import {
  getDynamicEvaluationRuns,
  getDynamicEvaluationRunEpisodesByRunIdWithFeedback,
  getDynamicEvaluationRunStatisticsByMetricName,
  countDynamicEvaluationRunEpisodes,
} from "~/utils/clickhouse/dynamic_evaluations.server";
import BasicInfo from "./DynamicEvaluationRunBasicInfo";
import DynamicEvaluationRunEpisodesTable from "./DynamicEvaluationRunEpisodesTable";
import { logger } from "~/utils/logger";

export const handle: RouteHandle = {
  crumb: (match) => [
    "Runs",
    { label: match.params.run_id!, isIdentifier: true },
  ],
};

export async function loader({ request, params }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const run_id = params.run_id;
  const searchParams = new URLSearchParams(url.search);
  const offset = parseInt(searchParams.get("offset") || "0");
  const pageSize = parseInt(searchParams.get("pageSize") || "15");
  const [
    dynamicEvaluationRuns,
    dynamicEvaluationRunEpisodes,
    count,
    statistics,
  ] = await Promise.all([
    getDynamicEvaluationRuns(5, 0, run_id),
    getDynamicEvaluationRunEpisodesByRunIdWithFeedback(
      pageSize,
      offset,
      run_id,
    ),
    countDynamicEvaluationRunEpisodes(run_id),
    getDynamicEvaluationRunStatisticsByMetricName(run_id),
  ]);
  if (dynamicEvaluationRuns.length != 1) {
    throw new Error(
      `Expected exactly one dynamic evaluation run, got ${dynamicEvaluationRuns.length}`,
    );
  }
  const dynamicEvaluationRun = dynamicEvaluationRuns[0];
  return {
    dynamicEvaluationRun,
    dynamicEvaluationRunEpisodes,
    statistics,
    count,
    offset,
    pageSize,
  };
}

export default function DynamicEvaluationRunSummaryPage({
  loaderData,
}: Route.ComponentProps) {
  const navigate = useNavigate();
  const {
    dynamicEvaluationRun,
    dynamicEvaluationRunEpisodes,
    statistics,
    count,
    offset,
    pageSize,
  } = loaderData;

  const handleNextPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset + pageSize));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };
  const handlePreviousPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset - pageSize));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <PageLayout>
      <PageHeader heading={`Dynamic Evaluation Run `} />
      <BasicInfo dynamicEvaluationRun={dynamicEvaluationRun} count={count} />
      <SectionLayout>
        <DynamicEvaluationRunEpisodesTable
          episodes={dynamicEvaluationRunEpisodes}
          statistics={statistics}
        />
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={offset <= 0}
          disableNext={offset + pageSize >= count}
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
