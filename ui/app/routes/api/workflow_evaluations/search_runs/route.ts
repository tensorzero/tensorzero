import type { Route } from "./+types/route";
import { abortableTimeout } from "~/utils/common";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const projectName = url.searchParams.get("project_name");
  if (!projectName) {
    return new Response("Missing project_name parameter", { status: 400 });
  }
  const query = url.searchParams.get("q") || "";

  const tensorZeroClient = getTensorZeroClient();
  const response = await tensorZeroClient.searchWorkflowEvaluationRuns(
    100,
    0,
    projectName,
    query,
  );
  return new Response(JSON.stringify(response.runs), {
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
