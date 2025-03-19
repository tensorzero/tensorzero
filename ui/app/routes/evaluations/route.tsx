import { getEvalRunIds } from "~/utils/clickhouse/evaluation.server";
import type { Route } from "./+types/route";

export async function loader({ request, params }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const selected_eval_run_ids = searchParams.get("eval_run_ids");
  const selected_eval_run_ids_array = selected_eval_run_ids
    ? selected_eval_run_ids.split(",")
    : [];

  const available_eval_run_ids = await getEvalRunIds(params.eval_name);
}
