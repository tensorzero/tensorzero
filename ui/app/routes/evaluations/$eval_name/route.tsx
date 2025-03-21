import type { Route } from "./+types/route";
import { getConfig } from "~/utils/config/index.server";
import {
  getEvalStatistics,
  getEvalResults,
  getEvalRunIds,
  getEvaluatorMetricName,
} from "~/utils/clickhouse/evaluations.server";
import { EvaluationTable } from "./EvaluationTable";
import { PageLayout } from "~/components/layout/PageLayout";
import { ContentLayout } from "~/components/layout/ContentLayout";

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
    eval_name: params.eval_name,
    available_eval_run_ids,
    eval_results,
    eval_statistics,
  };
}

export default function EvaluationsPage({ loaderData }: Route.ComponentProps) {
  const { eval_name, available_eval_run_ids, eval_results, eval_statistics } =
    loaderData;

  return (
    <PageLayout
      title={`Evaluation: ${eval_name}`}
      description={`Results for evaluation "${eval_name}"`}
    >
      <ContentLayout>
        <EvaluationTable
          available_eval_run_ids={available_eval_run_ids}
          eval_results={eval_results}
          eval_statistics={eval_statistics}
        />
      </ContentLayout>
    </PageLayout>
  );
}
