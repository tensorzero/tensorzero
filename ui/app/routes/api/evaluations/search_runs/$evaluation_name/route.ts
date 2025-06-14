import { searchEvaluationRuns } from "~/utils/clickhouse/evaluations.server";
import { getConfig } from "~/utils/config/index.server";
import type { Route } from "./+types/route";
import { abortableTimeout } from "~/utils/common";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const evaluationName = url.searchParams.get("evaluation_name");
  if (!evaluationName) {
    return new Response("Missing evaluation_name parameter", { status: 400 });
  }
  const query = url.searchParams.get("q") || "";
  const config = await getConfig();
  const function_name = config.evaluations[evaluationName].function_name;

  if (!evaluationName) {
    return new Response("Missing evaluation_name parameter", { status: 400 });
  }

  const runs = await searchEvaluationRuns(
    evaluationName,
    function_name,
    query,
    100,
    0,
  );
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
