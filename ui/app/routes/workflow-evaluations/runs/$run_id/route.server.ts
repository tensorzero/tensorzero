import { data } from "react-router";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

type WorkflowEvaluationRun = Awaited<
  ReturnType<
    ReturnType<typeof getTensorZeroClient>["listWorkflowEvaluationRuns"]
  >
>["runs"][0];

export type BasicInfoData = {
  workflowEvaluationRun: WorkflowEvaluationRun;
  count: number;
};

export type EpisodesData = {
  episodes: Awaited<
    ReturnType<
      ReturnType<
        typeof getTensorZeroClient
      >["getWorkflowEvaluationRunEpisodesWithFeedback"]
    >
  >["episodes"];
  statistics: Awaited<
    ReturnType<
      ReturnType<
        typeof getTensorZeroClient
      >["getWorkflowEvaluationRunStatistics"]
    >
  >["statistics"];
  count: number;
};

export async function fetchRunRecord(
  run_id: string,
): Promise<WorkflowEvaluationRun> {
  const client = getTensorZeroClient();
  const runsResponse = await client.listWorkflowEvaluationRuns(5, 0, run_id);

  const runs = runsResponse.runs;
  if (runs.length !== 1) {
    throw data(`Workflow evaluation run "${run_id}" not found`, {
      status: 404,
    });
  }

  return runs[0];
}

export async function fetchEpisodesTableData(
  run_id: string,
  limit: number,
  offset: number,
): Promise<Omit<EpisodesData, "count">> {
  const client = getTensorZeroClient();
  const [episodesResponse, statisticsResponse] = await Promise.all([
    client.getWorkflowEvaluationRunEpisodesWithFeedback(run_id, limit, offset),
    client.getWorkflowEvaluationRunStatistics(run_id),
  ]);

  return {
    episodes: episodesResponse.episodes,
    statistics: statisticsResponse.statistics,
  };
}
