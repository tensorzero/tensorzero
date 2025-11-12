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
  getWorkflowEvaluationRuns,
  getWorkflowEvaluationRunEpisodesByRunIdWithFeedback,
  getWorkflowEvaluationRunStatisticsByMetricName,
  countWorkflowEvaluationRunEpisodes,
} from "~/utils/clickhouse/workflow_evaluations.server";
import BasicInfo from "./WorkflowEvaluationRunBasicInfo";
import WorkflowEvaluationRunEpisodesTable from "./WorkflowEvaluationRunEpisodesTable";
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
  const limit = parseInt(searchParams.get("limit") || "15");
  const [
    workflowEvaluationRuns,
    workflowEvaluationRunEpisodes,
    count,
    statistics,
  ] = await Promise.all([
    getWorkflowEvaluationRuns(5, 0, run_id),
    getWorkflowEvaluationRunEpisodesByRunIdWithFeedback(limit, offset, run_id),
    countWorkflowEvaluationRunEpisodes(run_id),
    getWorkflowEvaluationRunStatisticsByMetricName(run_id),
  ]);
  if (workflowEvaluationRuns.length != 1) {
    throw new Error(
      `Expected exactly one workflow evaluation run, got ${workflowEvaluationRuns.length}`,
    );
  }
  const workflowEvaluationRun = workflowEvaluationRuns[0];
  return {
    workflowEvaluationRun,
    workflowEvaluationRunEpisodes,
    statistics,
    count,
    offset,
    limit,
  };
}

export default function WorkflowEvaluationRunSummaryPage({
  loaderData,
}: Route.ComponentProps) {
  const navigate = useNavigate();
  const {
    workflowEvaluationRun,
    workflowEvaluationRunEpisodes,
    statistics,
    count,
    offset,
    limit,
  } = loaderData;

  const handleNextPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset + limit));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };
  const handlePreviousPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset - limit));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <PageLayout>
      <PageHeader heading={`Workflow Evaluation Run `} />
      <BasicInfo workflowEvaluationRun={workflowEvaluationRun} count={count} />
      <SectionLayout>
        <WorkflowEvaluationRunEpisodesTable
          episodes={workflowEvaluationRunEpisodes}
          statistics={statistics}
        />
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={offset <= 0}
          disableNext={offset + limit >= count}
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
