import type { Route } from "./+types/route";
import { abortableTimeout } from "~/utils/common";
import { getConfig } from "~/utils/config/index.server";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const evaluationName = url.searchParams.get("evaluation_name");
  if (!evaluationName) {
    return new Response("Missing evaluation_name parameter", { status: 400 });
  }
  const query = url.searchParams.get("q") || "";
  const config = await getConfig();
  const function_name = config.evaluations[evaluationName]?.function_name;

  const runs = await getTensorZeroClient()
    .searchEvaluationRuns(
      evaluationName,
      query,
      function_name,
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
