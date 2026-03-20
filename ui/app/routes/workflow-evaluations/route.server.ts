import { getTensorZeroClient } from "~/utils/tensorzero.server";

export type ProjectsTableData = {
  projects: Awaited<
    ReturnType<
      ReturnType<typeof getTensorZeroClient>["getWorkflowEvaluationProjects"]
    >
  >["projects"];
  count: number;
};

export type RunsTableData = {
  runs: Awaited<
    ReturnType<
      ReturnType<typeof getTensorZeroClient>["listWorkflowEvaluationRuns"]
    >
  >["runs"];
  count: number;
};

export async function fetchProjectsTableData(
  limit: number,
  offset: number,
  countPromise: Promise<number>,
): Promise<ProjectsTableData> {
  const client = getTensorZeroClient();
  const [response, count] = await Promise.all([
    client.getWorkflowEvaluationProjects(limit, offset),
    countPromise,
  ]);
  return { projects: response.projects, count };
}

export async function fetchRunsTableData(
  limit: number,
  offset: number,
  countPromise: Promise<number>,
): Promise<RunsTableData> {
  const client = getTensorZeroClient();
  const [response, count] = await Promise.all([
    client.listWorkflowEvaluationRuns(limit, offset),
    countPromise,
  ]);
  return { runs: response.runs, count };
}
