import type { Route } from "./+types/route";
import { getConfig } from "~/utils/config/index.server";
import {
  getEvalStatistics,
  getEvalResults,
  getEvalRunIds,
  countDatapointsForEval,
  getMostRecentEvalInferenceDate,
} from "~/utils/clickhouse/evaluations.server";
import { getEvaluatorMetricName } from "~/utils/clickhouse/evaluations";
import { EvaluationTable } from "./EvaluationTable";
import {
  PageHeader,
  PageLayout,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import PageButtons from "~/components/utils/PageButtons";
import { useNavigate } from "react-router";
import AutoRefreshIndicator, { useAutoRefresh } from "./AutoRefreshIndicator";
import BasicInfo from "./EvaluationBasicInfo";
import { useConfig } from "~/context/config";

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
  const offset = parseInt(searchParams.get("offset") || "0");
  const pageSize = parseInt(searchParams.get("pageSize") || "15");

  const evaluator_names = Object.keys(
    config.evals[params.eval_name].evaluators,
  );
  const metric_names = evaluator_names.map((evaluatorName) =>
    getEvaluatorMetricName(params.eval_name, evaluatorName),
  );

  // Set up all promises to run concurrently
  const evalRunIdsPromise = getEvalRunIds(params.eval_name);
  const mostRecentEvalInferenceDatePromise = getMostRecentEvalInferenceDate(
    selected_eval_run_ids_array,
  );

  // Create placeholder promises for results and statistics that will be used conditionally
  const resultsPromise =
    selected_eval_run_ids_array.length > 0
      ? getEvalResults(
          dataset_name,
          function_name,
          function_type,
          metric_names,
          selected_eval_run_ids_array,
          pageSize,
          offset,
        )
      : Promise.resolve([]);

  const statisticsPromise =
    selected_eval_run_ids_array.length > 0
      ? getEvalStatistics(
          dataset_name,
          function_name,
          function_type,
          metric_names,
          selected_eval_run_ids_array,
        )
      : Promise.resolve([]);
  const total_datapoints_promise =
    selected_eval_run_ids_array.length > 0
      ? countDatapointsForEval(
          dataset_name,
          function_name,
          function_type,
          selected_eval_run_ids_array,
        )
      : 0;

  // Wait for all three promises to complete concurrently
  const [
    available_eval_run_ids,
    eval_results,
    eval_statistics,
    total_datapoints,
    mostRecentEvalInferenceDates,
  ] = await Promise.all([
    evalRunIdsPromise,
    resultsPromise,
    statisticsPromise,
    total_datapoints_promise,
    mostRecentEvalInferenceDatePromise,
  ]);

  return {
    eval_name: params.eval_name,
    available_eval_run_ids,
    eval_results,
    eval_statistics,
    has_selected_runs: selected_eval_run_ids_array.length > 0,
    offset,
    pageSize,
    total_datapoints,
    evaluator_names,
    mostRecentEvalInferenceDates,
  };
}

export default function EvaluationsPage({ loaderData }: Route.ComponentProps) {
  const {
    eval_name,
    available_eval_run_ids,
    eval_results,
    eval_statistics,
    has_selected_runs,
    offset,
    pageSize,
    total_datapoints,
    evaluator_names,
    mostRecentEvalInferenceDates,
  } = loaderData;
  const navigate = useNavigate();
  const handleNextPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset + pageSize));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };
  const handlePreviousPage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("offset", String(offset - pageSize));
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  // Get the most recent inference date from the map
  const mostRecentEvalInferenceDate = Array.from(
    mostRecentEvalInferenceDates.values(),
  ).reduce((max, current) => (current > max ? current : max), new Date(0));
  // Use that time for auto-refreshing
  const isAutoRefreshing = useAutoRefresh(mostRecentEvalInferenceDate);

  const config = useConfig();
  const eval_config = config.evals[eval_name];

  return (
    <PageLayout>
      <PageHeader heading="Evaluation" name={eval_name}>
        <BasicInfo eval_config={eval_config} />
      </PageHeader>

      <SectionsGroup>
        <SectionLayout>
          <div className="flex items-center justify-between">
            <SectionHeader heading="Results" />
            <AutoRefreshIndicator isActive={isAutoRefreshing} />
          </div>

          <EvaluationTable
            eval_name={eval_name}
            available_eval_run_ids={available_eval_run_ids}
            eval_results={eval_results}
            eval_statistics={eval_statistics}
            evaluator_names={evaluator_names}
            mostRecentEvalInferenceDates={mostRecentEvalInferenceDates}
          />
          <PageButtons
            onPreviousPage={handlePreviousPage}
            onNextPage={handleNextPage}
            disablePrevious={offset <= 0}
            disableNext={offset + pageSize >= total_datapoints}
          />
          {!has_selected_runs && (
            <div className="mt-4 text-center text-gray-500">
              Select evaluation run IDs to view results
            </div>
          )}
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
