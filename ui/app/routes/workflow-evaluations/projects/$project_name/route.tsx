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
import type {
  WorkflowEvaluationRun,
  WorkflowEvaluationRunStatistics,
} from "~/types/tensorzero";
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

type ResultsData = {
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

async function fetchResultsData(
  runIds: string[],
  limit: number,
  offset: number,
): Promise<ResultsData> {
  const tensorZeroClient = getTensorZeroClient();

  const statsPromises = runIds.map((runId) =>
    tensorZeroClient
      .getWorkflowEvaluationRunStatistics(runId)
      .then((response) => response.statistics),
  );

  const episodeInfoPromise = tensorZeroClient
    .listWorkflowEvaluationRunEpisodesByTaskName(runIds, limit, offset)
    .then((response) => response.episodes);
  const countPromise =
    tensorZeroClient.countWorkflowEvaluationRunEpisodeGroupsByTaskName(runIds);

  const [statsResults, episodeInfo, count] = await Promise.all([
    Promise.all(statsPromises),
    episodeInfoPromise,
    countPromise,
  ]);

  const runStats: Record<string, WorkflowEvaluationRunStatistics[]> = {};
  runIds.forEach((runId, index) => {
    runStats[runId] = statsResults[index];
  });

  return {
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

  const tensorZeroClient = getTensorZeroClient();
  let runInfos: WorkflowEvaluationRun[] = [];
  if (runIds.length > 0) {
    const response = await tensorZeroClient.getWorkflowEvaluationRuns(
      runIds,
      projectName,
    );
    runInfos = response.runs;
    // Sort runInfos to match the order from URL params
    runInfos.sort((a, b) => runIds.indexOf(a.id) - runIds.indexOf(b.id));
  }

  return {
    projectName,
    runInfos,
    resultsData:
      runIds.length > 0
        ? fetchResultsData(runIds, limit, offset)
        : Promise.resolve(null),
    limit,
    offset,
  };
}

function ResultsSkeleton() {
  return <Skeleton className="h-64 w-full" />;
}

function ResultsError() {
  const error = useAsyncError();
  let message = "Failed to load workflow evaluation results";
  if (isRouteErrorResponse(error)) {
    message = typeof error.data === "string" ? error.data : message;
  } else if (error instanceof Error) {
    message = error.message;
  }
  return (
    <SectionErrorNotice
      icon={AlertCircle}
      title="Error loading results"
      description={message}
    />
  );
}

function ResultsContent({
  runInfos,
  data,
  limit,
  offset,
}: {
  runInfos: WorkflowEvaluationRun[];
  data: ResultsData;
  limit: number;
  offset: number;
}) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { runStats, episodeInfo, count } = data;

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
    <>
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
    </>
  );
}

export default function WorkflowEvaluationProjectPage({
  loaderData,
}: Route.ComponentProps) {
  const { projectName, runInfos, resultsData, limit, offset } = loaderData;
  const location = useLocation();
  const selectedRunIds = runInfos.map((run) => run.id);

  return (
    <ColorAssignerProvider selectedRunIds={selectedRunIds}>
      <PageLayout>
        <ProjectPageHeader projectName={projectName} />
        <SectionLayout>
          <WorkflowEvalRunSelector
            projectName={projectName}
            selectedRunInfos={runInfos}
          />
          <Suspense key={location.key} fallback={<ResultsSkeleton />}>
            <Await resolve={resultsData} errorElement={<ResultsError />}>
              {(data) =>
                data && (
                  <ResultsContent
                    runInfos={runInfos}
                    data={data}
                    limit={limit}
                    offset={offset}
                  />
                )
              }
            </Await>
          </Suspense>
        </SectionLayout>
      </PageLayout>
    </ColorAssignerProvider>
  );
}
