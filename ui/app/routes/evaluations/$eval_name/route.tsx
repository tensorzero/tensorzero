import type { Route } from "./+types/route";
import { getConfig } from "~/utils/config/index.server";
import {
  getEvalStatistics,
  getEvalResults,
  getEvalRunIds,
  getEvaluatorMetricName,
  countDatapointsForEval,
} from "~/utils/clickhouse/evaluations.server";
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

  const metric_names = Object.keys(
    config.evals[params.eval_name].evaluators,
  ).map((evaluatorName) =>
    getEvaluatorMetricName(params.eval_name, evaluatorName),
  );

  // Set up all three promises to run concurrently
  const evalRunIdsPromise = getEvalRunIds(params.eval_name);

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
  ] = await Promise.all([
    evalRunIdsPromise,
    resultsPromise,
    statisticsPromise,
    total_datapoints_promise,
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

  return (
    <PageLayout>
      <PageHeader heading="Evaluation" name={eval_name} />

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Results" />
          <EvaluationTable
            available_eval_run_ids={available_eval_run_ids}
            eval_results={eval_results}
            eval_statistics={eval_statistics}
          />
          <PageButtons
            onPreviousPage={handlePreviousPage}
            onNextPage={handleNextPage}
            disablePrevious={offset === 0}
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
