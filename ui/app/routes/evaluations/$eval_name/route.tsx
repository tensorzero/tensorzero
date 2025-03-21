import type { Route } from "./+types/route";
import { getConfig } from "~/utils/config/index.server";
import {
  getEvalStatistics,
  getEvalResults,
  getEvalRunIds,
  getEvaluatorMetricName,
} from "~/utils/clickhouse/evaluations.server";

export async function loader({ request, params }: Route.LoaderArgs) {
  const config = await getConfig();
  const dataset_name = config.evals[params.eval_name].dataset_name;
  const function_name = config.evals[params.eval_name].function_name;
  const function_type = config.functions[function_name].type;
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const selected_eval_run_ids = searchParams.get("eval_run_ids");
  const selected_eval_run_ids_array = selected_eval_run_ids
    ? selected_eval_run_ids.split(",")
    : [];

  const metric_names = Object.keys(
    config.evals[params.eval_name].evaluators,
  ).map((evaluatorName) =>
    getEvaluatorMetricName(params.eval_name, evaluatorName),
  );

  // Run these concurrently
  const [available_eval_run_ids, eval_results, eval_statistics] =
    await Promise.all([
      getEvalRunIds(params.eval_name),
      getEvalResults(
        dataset_name,
        function_name,
        function_type,
        metric_names,
        selected_eval_run_ids_array,
      ),
      getEvalStatistics(
        dataset_name,
        function_name,
        function_type,
        metric_names,
        selected_eval_run_ids_array,
      ),
    ]);
  return {
    available_eval_run_ids,
    eval_results,
    eval_statistics,
  };
}

export default function EvaluationsPage({ loaderData }: Route.ComponentProps) {
  const { available_eval_run_ids, eval_results, eval_statistics } = loaderData;
  return (
    <div>
      <h1>Evaluations</h1>
    </div>
  );
}
