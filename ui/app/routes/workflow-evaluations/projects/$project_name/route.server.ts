import type { WorkflowEvaluationRunStatistics } from "~/types/tensorzero";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export type ResultsData = {
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

export async function fetchResultsData(
  runIds: string[],
  projectName: string,
  limit: number,
  offset: number,
): Promise<ResultsData> {
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
