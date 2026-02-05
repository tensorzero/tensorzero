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
import { useNavigate, useSearchParams, type RouteHandle } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export const handle: RouteHandle = {
  crumb: (match) => [
    "Projects",
    { label: match.params.project_name!, isIdentifier: true },
  ],
};

async function fetchResultsData(
  runIds: string[],
  projectName: string,
  limit: number,
  offset: number,
) {
  const client = getTensorZeroClient();
  const statsPromises = runIds.map((runId) =>
    client
      .getWorkflowEvaluationRunStatistics(runId)
      .then((response) => response.statistics),
  );
  const runInfosPromise = client
    .getWorkflowEvaluationRuns(runIds, projectName)
    .then((response) => response.runs);
  const episodeInfoPromise = client
    .listWorkflowEvaluationRunEpisodesByTaskName(runIds, limit, offset)
    .then((response) => response.episodes);
  const countPromise =
    client.countWorkflowEvaluationRunEpisodeGroupsByTaskName(runIds);

  const [statsResults, runInfos, episodeInfo, count] = await Promise.all([
    Promise.all(statsPromises),
    runInfosPromise,
    episodeInfoPromise,
    countPromise,
  ]);

  const runStats: Record<string, WorkflowEvaluationRunStatistics[]> = {};
  runIds.forEach((runId, index) => {
    runStats[runId] = statsResults[index];
  });
  // Sort runInfos by the same order as the url params
  runInfos.sort((a, b) => runIds.indexOf(a.id) - runIds.indexOf(b.id));

  return { runInfos, runStats, episodeInfo, count };
}

export async function loader({ request, params }: Route.LoaderArgs) {
  const projectName = params.project_name;
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const limit = parseInt(searchParams.get("limit") || "15");
  const offset = parseInt(searchParams.get("offset") || "0");
  const runIds = searchParams.get("run_ids")?.split(",") || [];

  if (runIds.length > 0) {
    const { runInfos, runStats, episodeInfo, count } = await fetchResultsData(
      runIds,
      projectName,
      limit,
      offset,
    );

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

function ResultsContent({
  runInfos,
  runStats,
  episodeInfo,
  count,
  limit,
  offset,
}: {
  runInfos: Awaited<ReturnType<typeof fetchResultsData>>["runInfos"];
  runStats: Awaited<ReturnType<typeof fetchResultsData>>["runStats"];
  episodeInfo: Awaited<ReturnType<typeof fetchResultsData>>["episodeInfo"];
  count: number;
  limit: number;
  offset: number;
}) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

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
  const { projectName, runInfos, runStats, episodeInfo, count, limit, offset } =
    loaderData;
  const selectedRunIds = runInfos.map((run) => run.id);

  return (
    <ColorAssignerProvider selectedRunIds={selectedRunIds}>
      <PageLayout>
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
        <SectionLayout>
          <WorkflowEvalRunSelector
            projectName={projectName}
            selectedRunInfos={runInfos}
          />
          <ResultsContent
            runInfos={runInfos}
            runStats={runStats}
            episodeInfo={episodeInfo}
            count={count}
            limit={limit}
            offset={offset}
          />
        </SectionLayout>
      </PageLayout>
    </ColorAssignerProvider>
  );
}
