import { getConfig, getConfigForSnapshot } from "~/utils/config/index.server";
import type { Route } from "./+types/route";
import { abortableTimeout } from "~/utils/common";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const evaluationName = url.searchParams.get("evaluation_name");
  if (!evaluationName) {
    return new Response("Missing evaluation_name parameter", { status: 400 });
  }
  const query = url.searchParams.get("q") || "";
  const config = await getConfig();
  let function_name = config.evaluations[evaluationName]?.function_name;

  if (!function_name) {
    // Evaluation not in current config — try to find it from a historical snapshot
    const client = getTensorZeroClient();
    const runs = await client.listEvaluationRuns(100, 0);
    const matchingRun = runs.runs.find(
      (r) => r.evaluation_name === evaluationName,
    );

    if (matchingRun?.snapshot_hash) {
      const snapshotConfig = await getConfigForSnapshot(
        matchingRun.snapshot_hash,
      );
      function_name = snapshotConfig.evaluations[evaluationName]?.function_name;
    }
  }

  if (!function_name) {
    return new Response(
      `Failed to find config for evaluation ${evaluationName}`,
      { status: 400 },
    );
  }

  const runs = await getTensorZeroClient()
    .searchEvaluationRuns(
      evaluationName,
      function_name,
      query,
      /*limit=*/ 100,
      /*offset=*/ 0,
    )
    .then((response) => response.results);
  return new Response(JSON.stringify(runs), {
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
