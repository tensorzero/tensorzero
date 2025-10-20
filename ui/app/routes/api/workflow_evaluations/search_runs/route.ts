import { searchWorkflowEvaluationRuns } from "~/utils/clickhouse/workflow_evaluations.server";
import { workflowEvaluationRunSchema } from "~/utils/clickhouse/workflow_evaluations";
import type { Route } from "./+types/route";
import { abortableTimeout } from "~/utils/common";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const projectName = url.searchParams.get("project_name");
  if (!projectName) {
    return new Response("Missing project_name parameter", { status: 400 });
  }
  const query = url.searchParams.get("q") || "";

  const runs = await searchWorkflowEvaluationRuns(100, 0, projectName, query);
  const parsedRuns = runs.map((run) => workflowEvaluationRunSchema.parse(run));
  return new Response(JSON.stringify(parsedRuns), {
    headers: {
      "Content-Type": "application/json",
    },
  });
}

export async function clientLoader({
  request,
  serverLoader,
}: Route.ClientLoaderArgs) {
  const url = new URL(request.url);
  const debounce = url.searchParams.get("debounce");
  if (debounce !== null) {
    await abortableTimeout(request, 500);
  }
  return await serverLoader();
}
