import { Suspense } from "react";
import { AlertCircle } from "lucide-react";
import {
  PageHeader,
  SectionLayout,
  PageLayout,
  Breadcrumbs,
} from "~/components/layout/PageLayout";
import type { Route } from "./+types/route";
import { WorkflowEvalRunSelector } from "~/routes/workflow-evaluations/projects/$project_name/WorkflowEvalRunSelector";
import type { WorkflowEvaluationRunStatistics } from "~/types/tensorzero";
import { ColorAssignerProvider } from "~/hooks/evaluations/ColorAssigner";
import { WorkflowEvaluationProjectResultsTable } from "./WorkflowEvaluationProjectResultsTable";
import {
  Await,
  isRouteErrorResponse,
  useAsyncError,
  useLocation,
  useNavigate,
  useSearchParams,
  type RouteHandle,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import { Skeleton } from "~/components/ui/skeleton";
import { SectionErrorNotice } from "~/components/ui/error/ErrorContentPrimitives";

export const handle: RouteHandle = {
  crumb: (match) => [
    "Projects",
    { label: match.params.project_name!, isIdentifier: true },
  ],
};

function ProjectPageHeader({ projectName }: { projectName: string }) {
  return (
    <PageHeader
      eyebrow={
        <Breadcrumbs
          segments={[
            {
              label: "Workflow Evaluations",
              href: "/workflow-evaluations",
            },
            { label: "Projects" },
          ]}
        />
      }
      name={projectName}
    />
  );
}

type ProjectData = {
  runInfos: Awaited<
    ReturnType<
      ReturnType<typeof getTensorZeroClient>["getWorkflowEvaluationRuns"]
    >
  >["runs"];
  runStats: Record<string, WorkflowEvaluationRunStatistics[]>;
  episodeInfo: Awaited<
    ReturnType<
      ReturnType<
        typeof getTensorZeroClient
      >["listWorkflowEvaluationRunEpisodesByTaskName"]
    >
  >["episodes"];
  count: number;
};

async function fetchProjectData(
  projectName: string,
  runIds: string[],
  limit: number,
  offset: number,
): Promise<ProjectData> {
  if (runIds.length === 0) {
    return {
      runInfos: [],
      runStats: {},
      episodeInfo: [],
      count: 0,
    };
  }

  const runStats: Record<string, WorkflowEvaluationRunStatistics[]> = {};
  const tensorZeroClient = getTensorZeroClient();

  const statsPromises = runIds.map((runId) =>
    tensorZeroClient
      .getWorkflowEvaluationRunStatistics(runId)
      .then((response) => response.statistics),
  );

  const runInfosPromise = tensorZeroClient
    .getWorkflowEvaluationRuns(runIds, projectName)
    .then((response) => response.runs);

  const episodeInfoPromise = tensorZeroClient
    .listWorkflowEvaluationRunEpisodesByTaskName(runIds, limit, offset)
    .then((response) => response.episodes);
  const countPromise =
    tensorZeroClient.countWorkflowEvaluationRunEpisodeGroupsByTaskName(runIds);

  const [statsResults, runInfos, episodeInfo, count] = await Promise.all([
    Promise.all(statsPromises),
    runInfosPromise,
    episodeInfoPromise,
    countPromise,
  ]);

  runIds.forEach((runId, index) => {
    runStats[runId] = statsResults[index];
  });

  // Sort runInfos to match the order from URL params
  runInfos.sort((a, b) => runIds.indexOf(a.id) - runIds.indexOf(b.id));

  return {
    runInfos,
    runStats,
    episodeInfo,
    count,
  };
}

export async function loader({ request, params }: Route.LoaderArgs) {
  const projectName = params.project_name;
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const limit = parseInt(searchParams.get("limit") || "15");
  const offset = parseInt(searchParams.get("offset") || "0");
  const runIds = searchParams.get("run_ids")?.split(",") || [];

  return {
    projectName,
    projectData: fetchProjectData(projectName, runIds, limit, offset),
    limit,
    offset,
  };
}

function ContentSkeleton({ projectName }: { projectName: string }) {
  return (
    <>
      <ProjectPageHeader projectName={projectName} />
      <SectionLayout>
        <Skeleton className="mb-4 h-10 w-full" />
        <Skeleton className="h-64 w-full" />
      </SectionLayout>
    </>
  );
}

function ContentError({ projectName }: { projectName: string }) {
  const error = useAsyncError();
  let message = "Failed to load workflow evaluation project";
  if (isRouteErrorResponse(error)) {
    message = typeof error.data === "string" ? error.data : message;
  } else if (error instanceof Error) {
    message = error.message;
  }
  return (
    <>
      <ProjectPageHeader projectName={projectName} />
      <SectionErrorNotice
        icon={AlertCircle}
        title="Error loading workflow evaluation project"
        description={message}
      />
    </>
  );
}

function ProjectContent({
  projectName,
  data,
  limit,
  offset,
}: {
  projectName: string;
  data: ProjectData;
  limit: number;
  offset: number;
}) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { runInfos, runStats, episodeInfo, count } = data;
  const selectedRunIds = runInfos.map((run) => run.id);

  const handleNextPage = () => {
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set("offset", String(offset + limit));
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousPage = () => {
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set("offset", String(offset - limit));
    navigate(`?${newSearchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <ColorAssignerProvider selectedRunIds={selectedRunIds}>
      <ProjectPageHeader projectName={projectName} />
      <SectionLayout>
        <WorkflowEvalRunSelector
          projectName={projectName}
          selectedRunInfos={runInfos}
        />
        <WorkflowEvaluationProjectResultsTable
          selected_run_infos={runInfos}
          evaluation_results={episodeInfo}
          evaluation_statistics={runStats}
        />
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={offset <= 0}
          disableNext={offset + limit >= count}
        />
      </SectionLayout>
    </ColorAssignerProvider>
  );
}

export default function WorkflowEvaluationProjectPage({
  loaderData,
}: Route.ComponentProps) {
  const { projectName, projectData, limit, offset } = loaderData;
  const location = useLocation();

  return (
    <PageLayout>
      <Suspense
        key={location.key}
        fallback={<ContentSkeleton projectName={projectName} />}
      >
        <Await
          resolve={projectData}
          errorElement={<ContentError projectName={projectName} />}
        >
          {(data) => (
            <ProjectContent
              projectName={projectName}
              data={data}
              limit={limit}
              offset={offset}
            />
          )}
        </Await>
      </Suspense>
    </PageLayout>
  );
}
