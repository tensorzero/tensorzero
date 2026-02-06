import { getTensorZeroClient } from "~/utils/tensorzero.server";

export type ProjectsData = {
  projects: Awaited<
    ReturnType<
      ReturnType<typeof getTensorZeroClient>["getWorkflowEvaluationProjects"]
    >
  >["projects"];
  count: number;
};

export type RunsData = {
  runs: Awaited<
    ReturnType<
      ReturnType<typeof getTensorZeroClient>["listWorkflowEvaluationRuns"]
    >
  >["runs"];
  count: number;
};

export async function fetchProjectsData(
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

export async function fetchRunsData(
  limit: number,
  offset: number,
): Promise<RunsData> {
  const client = getTensorZeroClient();
  const [response, count] = await Promise.all([
    client.listWorkflowEvaluationRuns(limit, offset),
    client.countWorkflowEvaluationRuns(),
  ]);
  return { runs: response.runs, count };
}
