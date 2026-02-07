import type { WorkflowEvaluationRunStatistics } from "~/types/tensorzero";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export type ResultsData = {
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
  limit: number,
  offset: number,
): Promise<ResultsData> {
  const client = getTensorZeroClient();
  const statsPromises = runIds.map((runId) =>
    client
      .getWorkflowEvaluationRunStatistics(runId)
      .then((response) => response.statistics),
  );
  const episodeInfoPromise = client
    .listWorkflowEvaluationRunEpisodesByTaskName(runIds, limit, offset)
    .then((response) => response.episodes);
  const countPromise =
    client.countWorkflowEvaluationRunEpisodeGroupsByTaskName(runIds);

  const [statsResults, episodeInfo, count] = await Promise.all([
    Promise.all(statsPromises),
    episodeInfoPromise,
    countPromise,
  ]);

  const runStats: Record<string, WorkflowEvaluationRunStatistics[]> = {};
  runIds.forEach((runId, index) => {
    runStats[runId] = statsResults[index];
  });

  return { runStats, episodeInfo, count };
}
