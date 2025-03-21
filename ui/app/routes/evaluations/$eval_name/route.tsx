import type { Route } from "./+types/route";
import { getConfig } from "~/utils/config/index.server";
import {
  getEvalStatistics,
  getEvalResults,
  getEvalRunIds,
  getEvaluatorMetricName,
  type EvaluationStatistics,
} from "~/utils/clickhouse/evaluations.server";
import { EvaluationTable } from "./EvaluationTable";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import type { EvaluationResult } from "~/utils/clickhouse/evaluations";

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

  // Always fetch available run IDs
  const available_eval_run_ids = await getEvalRunIds(params.eval_name);

  // Only fetch results and statistics if run IDs are selected
  let eval_results: EvaluationResult[] = [];
  let eval_statistics: EvaluationStatistics[] = [];

  if (selected_eval_run_ids_array.length > 0) {
    [eval_results, eval_statistics] = await Promise.all([
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
  }

  return {
    eval_name: params.eval_name,
    available_eval_run_ids,
    eval_results,
    eval_statistics,
    has_selected_runs: selected_eval_run_ids_array.length > 0,
  };
}

export default function EvaluationsPage({ loaderData }: Route.ComponentProps) {
  const {
    eval_name,
    available_eval_run_ids,
    eval_results,
    eval_statistics,
    has_selected_runs,
  } = loaderData;

  return (
    <PageLayout>
      <PageHeader heading="Evaluation" name={eval_name} />

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Results" />
          {has_selected_runs ? (
            <EvaluationTable
              available_eval_run_ids={available_eval_run_ids}
              eval_results={eval_results}
              eval_statistics={eval_statistics}
            />
          ) : (
            <div className="py-4 text-center text-gray-500">
              Select evaluation run IDs to view results
            </div>
          )}
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
