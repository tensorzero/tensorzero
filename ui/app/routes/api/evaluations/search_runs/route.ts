import type { Route } from "./+types/route";
import { abortableTimeout } from "~/utils/common";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const functionName = url.searchParams.get("function_name");
  if (!functionName) {
    return new Response("Missing function_name parameter", { status: 400 });
  }

  const evaluationName = url.searchParams.get("evaluation_name") || undefined;
  const query = url.searchParams.get("q") || "";

  const runs = await getTensorZeroClient()
    .searchEvaluationRuns(
      query,
      functionName,
      evaluationName,
      /* limit= */ 100,
      /* offset= */ 0,
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
