import { PageHeader, SectionLayout } from "~/components/layout/PageLayout";
import { PageLayout } from "~/components/layout/PageLayout";
import type { Route } from "./+types/route";
import { WorkflowEvalRunSelector } from "~/routes/workflow_evaluations/projects/$project_name/WorkflowEvalRunSelector";
import type { WorkflowEvaluationRunStatistics } from "~/types/tensorzero";
import { ColorAssignerProvider } from "~/hooks/evaluations/ColorAssigner";
import { WorkflowEvaluationProjectResultsTable } from "./WorkflowEvaluationProjectResultsTable";
import { useNavigate, type RouteHandle } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export const handle: RouteHandle = {
  crumb: (match) => [
    "Projects",
    { label: match.params.project_name!, isIdentifier: true },
  ],
};

export async function loader({ request, params }: Route.LoaderArgs) {
  const projectName = params.project_name;
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const limit = parseInt(searchParams.get("limit") || "15");
  const offset = parseInt(searchParams.get("offset") || "0");
  const runIds = searchParams.get("run_ids")?.split(",") || [];

  const runStats: Record<string, WorkflowEvaluationRunStatistics[]> = {};

  const tensorZeroClient = getTensorZeroClient();
  if (runIds.length > 0) {
    // Create promises for fetching statistics for each runId
    const statsPromises = runIds.map((runId) =>
      tensorZeroClient
        .getWorkflowEvaluationRunStatistics(runId)
        .then((response) => response.statistics),
    );

    // Create promise for fetching run info
    const runInfosPromise = tensorZeroClient
      .getWorkflowEvaluationRuns(runIds, projectName)
      .then((response) => response.runs);

    const client = getTensorZeroClient();
    const episodeInfoPromise = client
      .listWorkflowEvaluationRunEpisodesByTaskName(runIds, limit, offset)
      .then((response) => response.episodes);
    const countPromise =
      client.countWorkflowEvaluationRunEpisodeGroupsByTaskName(runIds);
    // Run all promises concurrently
    const [statsResults, runInfos, episodeInfo, count] = await Promise.all([
      Promise.all(statsPromises),
      runInfosPromise,
      episodeInfoPromise,
      countPromise,
    ]);

    runIds.forEach((runId, index) => {
      runStats[runId] = statsResults[index];
    });
    // Sort runInfos by the same order as the url params
    runInfos.sort((a, b) => runIds.indexOf(a.id) - runIds.indexOf(b.id));

    return {
      projectName,
      runInfos,
      runStats,
      episodeInfo,
      count,
      limit,
      offset,
    };
  } else {
    return {
      projectName,
      runInfos: [],
      runStats: {},
      episodeInfo: [],
      count: 0,
      limit,
      offset,
    };
  }
}

export default function WorkflowEvaluationProjectPage({
  loaderData,
}: Route.ComponentProps) {
  const { projectName, runInfos, runStats, episodeInfo, count, limit, offset } =
    loaderData;
  const navigate = useNavigate();
  const selectedRunIds = runInfos.map((run) => run.id);
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
    <ColorAssignerProvider selectedRunIds={selectedRunIds}>
      <PageLayout>
        <PageHeader label="Workflow Evaluation Project" name={projectName} />
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
      </PageLayout>
    </ColorAssignerProvider>
  );
}
