import type { Route } from "./+types/route";
import { isRouteErrorResponse, useNavigate } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
} from "~/components/layout/PageLayout";
import WorkflowEvaluationRunsTable from "./WorkflowEvaluationRunsTable";
import WorkflowEvaluationProjectsTable from "./WorkflowEvaluationProjectsTable";
import { logger } from "~/utils/logger";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const runOffset = parseInt(searchParams.get("runOffset") || "0");
  const runLimit = parseInt(searchParams.get("runLimit") || "15");
  const projectOffset = parseInt(searchParams.get("projectOffset") || "0");
  const projectLimit = parseInt(searchParams.get("projectLimit") || "15");
  const tensorZeroClient = getTensorZeroClient();
  const [
    workflowEvaluationRunsResponse,
    count,
    workflowEvaluationProjectsResponse,
    projectCount,
  ] = await Promise.all([
    tensorZeroClient.listWorkflowEvaluationRuns(runLimit, runOffset),
    tensorZeroClient.countWorkflowEvaluationRuns(),
    tensorZeroClient.getWorkflowEvaluationProjects(projectLimit, projectOffset),
    tensorZeroClient.countWorkflowEvaluationProjects(),
  ]);
  const workflowEvaluationRuns = workflowEvaluationRunsResponse.runs;
  const workflowEvaluationProjects =
    workflowEvaluationProjectsResponse.projects;

  return {
    workflowEvaluationRuns,
    count,
    workflowEvaluationProjects,
    projectCount,
    runOffset,
    runLimit,
    projectOffset,
    projectLimit,
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
    runLimit,
    projectOffset,
    projectLimit,
  } = loaderData;
  const handleNextRunPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("runOffset", String(runOffset + runLimit));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };
  const handlePreviousRunPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("runOffset", String(runOffset - runLimit));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handleNextProjectPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("projectOffset", String(projectOffset + projectLimit));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousProjectPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("projectOffset", String(projectOffset - projectLimit));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <PageLayout>
      <PageHeader
        heading="Workflow Evaluations"
        subheading="End-to-end testing for multi-step LLM workflows"
      />
      <SectionLayout>
        <SectionHeader heading="Projects" count={projectCount} />
        <WorkflowEvaluationProjectsTable
          workflowEvaluationProjects={workflowEvaluationProjects}
        />
        <PageButtons
          onPreviousPage={handlePreviousProjectPage}
          onNextPage={handleNextProjectPage}
          disablePrevious={projectOffset <= 0}
          disableNext={projectOffset + projectLimit >= projectCount}
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
          disableNext={runOffset + runLimit >= count}
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
