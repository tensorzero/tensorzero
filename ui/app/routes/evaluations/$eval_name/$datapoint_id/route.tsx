import { getEvalsForDatapoint } from "~/utils/clickhouse/evaluations.server";
import type { Route } from "./+types/route";
import { getConfig } from "~/utils/config/index.server";
import {
  PageHeader,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { PageLayout } from "~/components/layout/PageLayout";
import Input from "~/components/inference/Input";
import { data, isRouteErrorResponse, redirect } from "react-router";
import Output from "~/components/inference/Output";
import {
  consolidate_eval_results,
  getEvaluatorMetricName,
  type ConsolidatedMetric,
} from "~/utils/clickhouse/evaluations";
import { useConfig } from "~/context/config";
import MetricValue from "~/components/evaluations/MetricValue";
import { getMetricType, type EvaluatorConfig } from "~/utils/config/evals";
import EvalRunBadge from "~/components/evaluations/EvalRunBadge";
import BasicInfo from "./EvaluationDatapointBasicInfo";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

export async function loader({ request, params }: Route.LoaderArgs) {
  const config = await getConfig();
  const eval_name = params.eval_name;
  const datapoint_id = params.datapoint_id;
  const dataset_name = config.evals[eval_name].dataset_name;
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);

  const selected_eval_run_ids = searchParams.get("eval_run_ids");
  const selected_eval_run_ids_array = selected_eval_run_ids
    ? selected_eval_run_ids.split(",")
    : [];
  if (selected_eval_run_ids_array.length === 0) {
    return redirect(`/datasets/${dataset_name}/datapoint/${datapoint_id}`);
  }
  const evalResults = await getEvalsForDatapoint(
    eval_name,
    datapoint_id,
    selected_eval_run_ids_array,
  );
  const consolidatedEvalResults = consolidate_eval_results(evalResults);
  if (consolidatedEvalResults.length !== selected_eval_run_ids_array.length) {
    // Find which eval run IDs are missing from the results
    const foundEvalRunIds = new Set(
      consolidatedEvalResults.map((result) => result.eval_run_id),
    );
    const missingEvalRunIds = selected_eval_run_ids_array.filter(
      (id) => !foundEvalRunIds.has(id),
    );

    throw data(
      `Evaluation run ID(s) not found: ${missingEvalRunIds.join(", ")}`,
      { status: 404 },
    );
  }
  return {
    consolidatedEvalResults,
    eval_name,
    datapoint_id,
  };
}

export default function EvaluationDatapointPage({
  loaderData,
}: Route.ComponentProps) {
  const { consolidatedEvalResults, eval_name, datapoint_id } = loaderData;
  const config = useConfig();
  const eval_config = config.evals[eval_name];
  const outputsToDisplay = [
    {
      id: "Reference",
      output: consolidatedEvalResults[0].reference_output,
      metrics: [],
      variant_name: "Reference",
    },
    ...consolidatedEvalResults.map((result) => ({
      id: result.eval_run_id,
      variant_name: result.variant_name,
      output: result.generated_output,
      metrics: result.metrics,
    })),
  ];

  // Function to get color for each run
  const getColor = () => {
    // Return a fixed color since we don't need dynamic colors
    return "bg-blue-500 hover:bg-blue-600 text-white";
  };

  return (
    <PageLayout>
      <PageHeader label="Datapoint" name={datapoint_id}>
        <BasicInfo eval_name={eval_name} eval_config={eval_config} />
      </PageHeader>

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Input" />
          <Input input={consolidatedEvalResults[0].input} />
        </SectionLayout>
        <SectionLayout>
          <SectionHeader heading="Output" />
          <div className="flex gap-4 overflow-x-auto pb-4">
            {outputsToDisplay.map((result) => (
              <div
                key={result.id}
                className="flex min-w-[300px] max-w-[450px] flex-shrink-0 flex-col justify-between"
              >
                <div>
                  <div className="mb-2 flex">
                    {result.id === "Reference" ? (
                      <EvalRunBadge
                        runInfo={{
                          eval_run_id: "",
                          variant_name: result.variant_name,
                        }}
                        getColor={() => "bg-gray-100 text-gray-700"}
                      />
                    ) : (
                      <EvalRunBadge
                        runInfo={{
                          eval_run_id: result.id,
                          variant_name: result.variant_name,
                        }}
                        getColor={getColor}
                      />
                    )}
                  </div>
                  <Output output={result.output} />
                </div>
                {result.id !== "Reference" &&
                  result.metrics &&
                  result.metrics.length > 0 && (
                    <MetricsDisplay
                      eval_name={eval_name}
                      metrics={result.metrics}
                      evaluatorsConfig={eval_config.evaluators}
                    />
                  )}
              </div>
            ))}
          </div>
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}

// Component to display metrics for a result
const MetricsDisplay = ({
  metrics,
  eval_name,
  evaluatorsConfig,
}: {
  metrics: ConsolidatedMetric[];
  eval_name: string;
  evaluatorsConfig: Record<string, EvaluatorConfig>;
}) => {
  return (
    <div className="mt-3 border-t border-gray-200 pt-2">
      <div className="space-y-1">
        {metrics.map((metricObj) => {
          const evaluatorConfig = evaluatorsConfig[metricObj.evaluator_name];
          if (!evaluatorConfig) return null;

          return (
            <MetricRow
              key={metricObj.evaluator_name}
              eval_name={eval_name}
              evaluatorName={metricObj.evaluator_name}
              metricValue={metricObj.metric_value}
              evaluatorConfig={evaluatorConfig}
            />
          );
        })}
      </div>
    </div>
  );
};

// Component for a single metric row
const MetricRow = ({
  evaluatorName,
  eval_name,
  metricValue,
  evaluatorConfig,
}: {
  evaluatorName: string;
  eval_name: string;
  metricValue: string;
  evaluatorConfig: EvaluatorConfig;
}) => {
  const config = useConfig();
  const metric_name = getEvaluatorMetricName(eval_name, evaluatorName);
  const metricProperties = config.metrics[metric_name];
  if (
    metricProperties.type === "comment" ||
    metricProperties.type === "demonstration"
  ) {
    return null;
  }
  return (
    <div className="flex items-center gap-2">
      <TooltipProvider delayDuration={300}>
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="cursor-help text-sm text-gray-600">
              {evaluatorName}:
            </div>
          </TooltipTrigger>
          <TooltipContent side="top" className="p-3">
            <div className="space-y-1 text-left text-xs">
              <div>
                <span className="font-medium">Type:</span>
                <span className="ml-2 font-medium">
                  {metricProperties.type}
                </span>
              </div>
              <div>
                <span className="font-medium">Optimize:</span>
                <span className="ml-2 font-medium">
                  {metricProperties.optimize}
                </span>
              </div>
              {evaluatorConfig.cutoff !== undefined && (
                <div>
                  <span className="font-medium">Cutoff:</span>
                  <span className="ml-2 font-medium">
                    {evaluatorConfig.cutoff}
                  </span>
                </div>
              )}
            </div>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <MetricValue
        value={String(metricValue)}
        metricType={getMetricType(evaluatorConfig)}
        evaluatorConfig={evaluatorConfig}
        className="text-sm"
      />
    </div>
  );
};

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  console.error(error);

  if (isRouteErrorResponse(error)) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">
          {error.status} {error.statusText}
        </h1>
        <p>{error.data}</p>
      </div>
    );
  } else if (error instanceof Error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">Error</h1>
        <p>{error.message}</p>
      </div>
    );
  } else {
    return (
      <div className="flex h-screen items-center justify-center text-red-500">
        <h1 className="text-2xl font-bold">Unknown Error</h1>
      </div>
    );
  }
}
