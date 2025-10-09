import { PageHeader, SectionLayout } from "~/components/layout/PageLayout";
import { PageLayout } from "~/components/layout/PageLayout";
import type { Route } from "./+types/route";
import { DynamicEvalRunSelector } from "~/routes/dynamic_evaluations/projects/$project_name/DynamicEvalRunSelector";
import {
  countDynamicEvaluationRunEpisodesByTaskName,
  getDynamicEvaluationRunEpisodesByTaskName,
  getDynamicEvaluationRunsByIds,
  getDynamicEvaluationRunStatisticsByMetricName,
} from "~/utils/clickhouse/dynamic_evaluations.server";
import type { DynamicEvaluationRunStatisticsByMetricName } from "~/utils/clickhouse/dynamic_evaluations";
import { ColorAssignerProvider } from "~/hooks/evaluations/ColorAssigner";
import { DynamicEvaluationProjectResultsTable } from "./DynamicEvaluationProjectResultsTable";
import { useNavigate, type RouteHandle } from "react-router";
import PageButtons from "~/components/utils/PageButtons";

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
  const pageSize = parseInt(searchParams.get("pageSize") || "15");
  const offset = parseInt(searchParams.get("offset") || "0");
  const runIds = searchParams.get("run_ids")?.split(",") || [];

  const runStats: Record<string, DynamicEvaluationRunStatisticsByMetricName[]> =
    {};

  if (runIds.length > 0) {
    // Create promises for fetching statistics for each runId
    const statsPromises = runIds.map((runId) =>
      getDynamicEvaluationRunStatisticsByMetricName(runId),
    );

    // Create promise for fetching run info
    const runInfosPromise = getDynamicEvaluationRunsByIds(runIds, projectName);

    const episodeInfoPromise = getDynamicEvaluationRunEpisodesByTaskName(
      runIds,
      pageSize,
      offset,
    );
    const countPromise = countDynamicEvaluationRunEpisodesByTaskName(runIds);
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
      pageSize,
      offset,
    };
  } else {
    return {
      projectName,
      runInfos: [],
      runStats: {},
      episodeInfo: [],
      count: 0,
      pageSize,
      offset,
    };
  }
}

export default function DynamicEvaluationProjectPage({
  loaderData,
}: Route.ComponentProps) {
  const {
    projectName,
    runInfos,
    runStats,
    episodeInfo,
    count,
    pageSize,
    offset,
  } = loaderData;
  const navigate = useNavigate();
  const selectedRunIds = runInfos.map((run) => run.id);
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
    <ColorAssignerProvider selectedRunIds={selectedRunIds}>
      <PageLayout>
        <PageHeader heading="Dynamic Evaluation Project" name={projectName} />
        <SectionLayout>
          <DynamicEvalRunSelector
            projectName={projectName}
            selectedRunInfos={runInfos}
          />
          <DynamicEvaluationProjectResultsTable
            selected_run_infos={runInfos}
            evaluation_results={episodeInfo}
            evaluation_statistics={runStats}
          />
          <PageButtons
            onPreviousPage={handlePreviousPage}
            onNextPage={handleNextPage}
            disablePrevious={offset <= 0}
            disableNext={offset + pageSize >= count}
          />
        </SectionLayout>
      </PageLayout>
    </ColorAssignerProvider>
  );
}
