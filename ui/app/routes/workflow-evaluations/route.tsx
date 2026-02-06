import { Suspense } from "react";
import { AlertCircle } from "lucide-react";
import type { Route } from "./+types/route";
import {
  Await,
  useAsyncError,
  useLocation,
  useNavigate,
  useSearchParams,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
} from "~/components/layout/PageLayout";
import WorkflowEvaluationRunsTable from "./WorkflowEvaluationRunsTable";
import WorkflowEvaluationProjectsTable from "./WorkflowEvaluationProjectsTable";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { Skeleton } from "~/components/ui/skeleton";
import {
  getErrorMessage,
  SectionErrorNotice,
} from "~/components/ui/error/ErrorContentPrimitives";

type ProjectsData = {
  projects: Awaited<
    ReturnType<
      ReturnType<typeof getTensorZeroClient>["getWorkflowEvaluationProjects"]
    >
  >["projects"];
  count: number;
};

type RunsData = {
  runs: Awaited<
    ReturnType<
      ReturnType<typeof getTensorZeroClient>["listWorkflowEvaluationRuns"]
    >
  >["runs"];
  count: number;
};

async function fetchProjectsData(
  limit: number,
  offset: number,
): Promise<ProjectsData> {
  const client = getTensorZeroClient();
  const [response, count] = await Promise.all([
    client.getWorkflowEvaluationProjects(limit, offset),
    client.countWorkflowEvaluationProjects(),
  ]);
  return { projects: response.projects, count };
}

async function fetchRunsData(limit: number, offset: number): Promise<RunsData> {
  const client = getTensorZeroClient();
  const [response, count] = await Promise.all([
    client.listWorkflowEvaluationRuns(limit, offset),
    client.countWorkflowEvaluationRuns(),
  ]);
  return { runs: response.runs, count };
}

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const runOffset = parseInt(searchParams.get("runOffset") || "0");
  const runLimit = parseInt(searchParams.get("runLimit") || "15");
  const projectOffset = parseInt(searchParams.get("projectOffset") || "0");
  const projectLimit = parseInt(searchParams.get("projectLimit") || "15");

  return {
    projectsData: fetchProjectsData(projectLimit, projectOffset),
    runsData: fetchRunsData(runLimit, runOffset),
    runOffset,
    runLimit,
    projectOffset,
    projectLimit,
  };
}

function SectionSkeleton() {
  return (
    <>
      <Skeleton className="mb-2 h-6 w-32" />
      <Skeleton className="h-48 w-full" />
    </>
  );
}

function SectionError({
  title,
  defaultMessage,
}: {
  title: string;
  defaultMessage: string;
}) {
  const error = useAsyncError();
  return (
    <SectionErrorNotice
      icon={AlertCircle}
      title={title}
      description={getErrorMessage({ error, fallback: defaultMessage })}
    />
  );
}

function ProjectsContent({
  data,
  offset,
  limit,
}: {
  data: ProjectsData;
  offset: number;
  limit: number;
}) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { projects, count } = data;

  const handleNextPage = () => {
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set("projectOffset", String(offset + limit));
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousPage = () => {
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set("projectOffset", String(offset - limit));
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <>
      <SectionHeader heading="Projects" count={count} />
      <WorkflowEvaluationProjectsTable workflowEvaluationProjects={projects} />
      <PageButtons
        onPreviousPage={handlePreviousPage}
        onNextPage={handleNextPage}
        disablePrevious={offset <= 0}
        disableNext={offset + limit >= count}
      />
    </>
  );
}

function RunsContent({
  data,
  offset,
  limit,
}: {
  data: RunsData;
  offset: number;
  limit: number;
}) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { runs, count } = data;

  const handleNextPage = () => {
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set("runOffset", String(offset + limit));
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousPage = () => {
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set("runOffset", String(offset - limit));
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <>
      <SectionHeader heading="Evaluation Runs" count={count} />
      <WorkflowEvaluationRunsTable workflowEvaluationRuns={runs} />
      <PageButtons
        onPreviousPage={handlePreviousPage}
        onNextPage={handleNextPage}
        disablePrevious={offset <= 0}
        disableNext={offset + limit >= count}
      />
    </>
  );
}

export default function EvaluationSummaryPage({
  loaderData,
}: Route.ComponentProps) {
  const {
    projectsData,
    runsData,
    runOffset,
    runLimit,
    projectOffset,
    projectLimit,
  } = loaderData;
  const location = useLocation();

  return (
    <PageLayout>
      <PageHeader heading="Workflow Evaluations" />
      <SectionLayout>
        <Suspense
          key={`projects-${location.key}`}
          fallback={<SectionSkeleton />}
        >
          <Await
            resolve={projectsData}
            errorElement={
              <SectionError
                title="Error loading projects"
                defaultMessage="Failed to load projects"
              />
            }
          >
            {(data) => (
              <ProjectsContent
                data={data}
                offset={projectOffset}
                limit={projectLimit}
              />
            )}
          </Await>
        </Suspense>
      </SectionLayout>
      <SectionLayout>
        <Suspense key={`runs-${location.key}`} fallback={<SectionSkeleton />}>
          <Await
            resolve={runsData}
            errorElement={
              <SectionError
                title="Error loading evaluation runs"
                defaultMessage="Failed to load evaluation runs"
              />
            }
          >
            {(data) => (
              <RunsContent data={data} offset={runOffset} limit={runLimit} />
            )}
          </Await>
        </Suspense>
      </SectionLayout>
    </PageLayout>
  );
}
