import type { Route } from "./+types/route";
import { isRouteErrorResponse, useNavigate } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
} from "~/components/layout/PageLayout";
import {
  getWorkflowEvaluationRuns,
  countWorkflowEvaluationRuns,
  getWorkflowEvaluationProjects,
  countWorkflowEvaluationProjects,
} from "~/utils/clickhouse/workflow_evaluations.server";
import WorkflowEvaluationRunsTable from "./WorkflowEvaluationRunsTable";
import WorkflowEvaluationProjectsTable from "./WorkflowEvaluationProjectsTable";
import { logger } from "~/utils/logger";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const runOffset = parseInt(searchParams.get("runOffset") || "0");
  const runPageSize = parseInt(searchParams.get("runPageSize") || "15");
  const projectOffset = parseInt(searchParams.get("projectOffset") || "0");
  const projectPageSize = parseInt(searchParams.get("projectPageSize") || "15");
  const [
    workflowEvaluationRuns,
    count,
    workflowEvaluationProjects,
    projectCount,
  ] = await Promise.all([
    getWorkflowEvaluationRuns(runPageSize, runOffset),
    countWorkflowEvaluationRuns(),
    getWorkflowEvaluationProjects(projectPageSize, projectOffset),
    countWorkflowEvaluationProjects(),
  ]);

  return {
    workflowEvaluationRuns,
    count,
    workflowEvaluationProjects,
    projectCount,
    runOffset,
    runPageSize,
    projectOffset,
    projectPageSize,
  };
}

export default function EvaluationSummaryPage({
  loaderData,
}: Route.ComponentProps) {
  const navigate = useNavigate();
  const {
    workflowEvaluationRuns,
    count,
    workflowEvaluationProjects,
    projectCount,
    runOffset,
    runPageSize,
    projectOffset,
    projectPageSize,
  } = loaderData;
  const handleNextRunPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("runOffset", String(runOffset + runPageSize));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };
  const handlePreviousRunPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("runOffset", String(runOffset - runPageSize));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handleNextProjectPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("projectOffset", String(projectOffset + projectPageSize));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousProjectPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("projectOffset", String(projectOffset - projectPageSize));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <PageLayout>
      <PageHeader heading="Workflow Evaluations" />
      <SectionLayout>
        <SectionHeader heading="Projects" count={projectCount} />
        <WorkflowEvaluationProjectsTable
          workflowEvaluationProjects={workflowEvaluationProjects}
        />
        <PageButtons
          onPreviousPage={handlePreviousProjectPage}
          onNextPage={handleNextProjectPage}
          disablePrevious={projectOffset <= 0}
          disableNext={projectOffset + projectPageSize >= projectCount}
        />
      </SectionLayout>
      <SectionLayout>
        <SectionHeader heading="Evaluation Runs" count={count} />
        <WorkflowEvaluationRunsTable
          workflowEvaluationRuns={workflowEvaluationRuns}
        />
        <PageButtons
          onPreviousPage={handlePreviousRunPage}
          onNextPage={handleNextRunPage}
          disablePrevious={runOffset <= 0}
          disableNext={runOffset + runPageSize >= count}
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
