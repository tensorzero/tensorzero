import { getEvalsForDatapoint } from "~/utils/clickhouse/evaluations.server";
import type { Route } from "./+types/route";
import { getConfig } from "~/utils/config/index.server";

export async function loader({ request, params }: Route.LoaderArgs) {
  const config = await getConfig();
  const eval_name = params.eval_name;
  const datapoint_id = params.datapoint_id;
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);

  const selected_eval_run_ids = searchParams.get("eval_run_ids");
  const selected_eval_run_ids_array = selected_eval_run_ids
    ? selected_eval_run_ids.split(",")
    : [];

  const evalResults = await getEvalsForDatapoint(
    eval_name,
    datapoint_id,
    selected_eval_run_ids_array,
  );
  return {
    evalResults,
    selected_eval_run_ids_array,
  };
}

export default function EvaluationDatapointPage({
  loaderData,
}: Route.ComponentProps) {
  const { eval_name, datapoint_id } = loaderData;
}
