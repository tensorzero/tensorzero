import { getEvaluationsForDatapoint } from "~/utils/clickhouse/evaluations.server";
import type { Route } from "./+types/route";
import {
  PageHeader,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { PageLayout } from "~/components/layout/PageLayout";
import Input from "~/components/inference/Input";
import { data, isRouteErrorResponse, redirect } from "react-router";
import Output from "~/components/inference/NewOutput";
import {
  consolidate_evaluation_results,
  getEvaluatorMetricName,
  type ConsolidatedMetric,
} from "~/utils/clickhouse/evaluations";
import { useConfig } from "~/context/config";
import MetricValue from "~/components/evaluations/MetricValue";
import {
  getMetricType,
  type EvaluatorConfig,
} from "~/utils/config/evaluations";
import EvaluationRunBadge from "~/components/evaluations/EvaluationRunBadge";
import BasicInfo from "./EvaluationDatapointBasicInfo";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

export async function loader({ request, params }: Route.LoaderArgs) {
  const evaluation_name = params.evaluation_name;
  const datapoint_id = params.datapoint_id;
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);

  const selected_evaluation_run_ids = searchParams.get("evaluation_run_ids");
  const selected_evaluation_run_ids_array = selected_evaluation_run_ids
    ? selected_evaluation_run_ids.split(",")
    : [];
  if (selected_evaluation_run_ids_array.length === 0) {
    return redirect(`/evaluations/${evaluation_name}`);
  }
  const EvaluationResults = await getEvaluationsForDatapoint(
    evaluation_name,
    datapoint_id,
    selected_evaluation_run_ids_array,
  );
  const consolidatedEvaluationResults =
    consolidate_evaluation_results(EvaluationResults);
  if (
    consolidatedEvaluationResults.length !==
    selected_evaluation_run_ids_array.length
  ) {
    // Find which evaluation run IDs are missing from the results
    const foundEvaluationRunIds = new Set(
      consolidatedEvaluationResults.map((result) => result.evaluation_run_id),
    );
    const missingEvaluationRunIds = selected_evaluation_run_ids_array.filter(
      (id) => !foundEvaluationRunIds.has(id),
    );

    throw data(
      `Evaluation run ID(s) not found: ${missingEvaluationRunIds.join(", ")}`,
      { status: 404 },
    );
  }
  return {
    consolidatedEvaluationResults,
    evaluation_name,
    datapoint_id,
  };
}

export default function EvaluationDatapointPage({
  loaderData,
}: Route.ComponentProps) {
  const { consolidatedEvaluationResults, evaluation_name, datapoint_id } =
    loaderData;
  const config = useConfig();
  const evaluation_config = config.evaluations[evaluation_name];
  const outputsToDisplay = [
    {
      id: "Reference",
      output: consolidatedEvaluationResults[0].reference_output,
      metrics: [],
      variant_name: "Reference",
    },
    ...consolidatedEvaluationResults.map((result) => ({
      id: result.evaluation_run_id,
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
        <BasicInfo
          evaluation_name={evaluation_name}
          evaluation_config={evaluation_config}
          dataset_name={consolidatedEvaluationResults[0].dataset_name}
        />
      </PageHeader>

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Input" />
          <Input input={consolidatedEvaluationResults[0].input} />
        </SectionLayout>
        <SectionLayout>
          <SectionHeader heading="Output" />
          <div className="flex gap-4 overflow-x-auto pb-4">
            {outputsToDisplay.map((result) => (
              <div
                key={result.id}
                className="flex max-w-[450px] min-w-[300px] shrink-0 flex-col justify-between"
              >
                <div>
                  <div className="mb-2 flex">
                    {result.id === "Reference" ? (
                      <EvaluationRunBadge
                        runInfo={{
                          evaluation_run_id: "",
                          variant_name: result.variant_name,
                        }}
                        getColor={() => "bg-gray-100 text-gray-700"}
                      />
                    ) : (
                      <EvaluationRunBadge
                        runInfo={{
                          evaluation_run_id: result.id,
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
                      evaluation_name={evaluation_name}
                      metrics={result.metrics}
                      evaluatorsConfig={evaluation_config.evaluators}
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
  evaluation_name,
  evaluatorsConfig,
}: {
  metrics: ConsolidatedMetric[];
  evaluation_name: string;
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
              evaluation_name={evaluation_name}
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
  evaluation_name,
  metricValue,
  evaluatorConfig,
}: {
  evaluatorName: string;
  evaluation_name: string;
  metricValue: string;
  evaluatorConfig: EvaluatorConfig;
}) => {
  const config = useConfig();
  const metric_name = getEvaluatorMetricName(evaluation_name, evaluatorName);
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
