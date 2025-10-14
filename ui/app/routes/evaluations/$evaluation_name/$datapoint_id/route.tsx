import {
  getEvaluationRunInfos,
  getEvaluationRunInfosForDatapoint,
  getEvaluationsForDatapoint,
  pollForEvaluations,
} from "~/utils/clickhouse/evaluations.server";
import { toEvaluationUrl, toInferenceUrl } from "~/utils/urls";
import type { Route } from "./+types/route";
import {
  PageHeader,
  SectionHeader,
  SectionLayout,
  SectionsGroup,
} from "~/components/layout/PageLayout";
import { PageLayout } from "~/components/layout/PageLayout";
import InputSnippet from "~/components/inference/InputSnippet";

import {
  data,
  isRouteErrorResponse,
  Link,
  redirect,
  useFetcher,
  type RouteHandle,
} from "react-router";
import { Output } from "~/components/inference/Output";
import {
  consolidate_evaluation_results,
  getEvaluatorMetricName,
  type ConsolidatedMetric,
} from "~/utils/clickhouse/evaluations";
import { useConfig } from "~/context/config";
import MetricValue from "~/components/metric/MetricValue";
import { getMetricType } from "~/utils/config/evaluations";
import EvaluationRunBadge from "~/components/evaluations/EvaluationRunBadge";
import BasicInfo from "./EvaluationDatapointBasicInfo";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { EvalRunSelector } from "~/components/evaluations/EvalRunSelector";
import {
  ColorAssignerProvider,
  useColorAssigner,
} from "~/hooks/evaluations/ColorAssigner";
import { getConfig } from "~/utils/config/index.server";
import type {
  EvaluationConfig,
  EvaluatorConfig,
  ContentBlockChatOutput,
  JsonInferenceOutput,
} from "tensorzero-node";
import EvaluationFeedbackEditor from "~/components/evaluations/EvaluationFeedbackEditor";
import { InferenceButton } from "~/components/utils/InferenceButton";
import { addEvaluationHumanFeedback } from "~/utils/tensorzero.server";
import { handleAddToDatasetAction } from "~/utils/dataset.server";
import { renameDatapoint } from "~/routes/datasets/$dataset_name/datapoint/$id/datapointOperations.server";
import { Toaster } from "~/components/ui/toaster";
import { useToast } from "~/hooks/use-toast";
import { useEffect } from "react";
import { AddToDatasetButton } from "~/components/dataset/AddToDatasetButton";
import { logger } from "~/utils/logger";
import { getDatapoint } from "~/utils/clickhouse/datasets.server";

export const handle: RouteHandle = {
  crumb: (match) => [
    "Datapoints",
    { label: match.params.datapoint_id!, isIdentifier: true },
  ],
};

export async function loader({ request, params }: Route.LoaderArgs) {
  const evaluation_name = params.evaluation_name;
  const datapoint_id = params.datapoint_id;
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const config = await getConfig();
  const evaluation_config = config.evaluations[evaluation_name];
  if (!evaluation_config) {
    throw data(
      `Evaluation config not found for evaluation ${evaluation_name}`,
      { status: 404 },
    );
  }
  const function_name = evaluation_config.function_name;
  const newFeedbackId = searchParams.get("newFeedbackId");
  const newJudgeDemonstrationId = searchParams.get("newJudgeDemonstrationId");

  const selected_evaluation_run_ids = searchParams.get("evaluation_run_ids");
  const selectedRunIds = selected_evaluation_run_ids
    ? selected_evaluation_run_ids.split(",")
    : [];
  if (selectedRunIds.length === 0) {
    return redirect(toEvaluationUrl(evaluation_name));
  }

  // Define all promises
  const selectedEvaluationRunInfosPromise = getEvaluationRunInfos(
    selectedRunIds,
    function_name,
  );
  const allowedEvaluationRunInfosPromise = getEvaluationRunInfosForDatapoint(
    datapoint_id,
    function_name,
  );

  // If there is a freshly inserted feedback, ClickHouse may take some time to
  // update the evaluation results as it is eventually consistent.
  // In this case, we poll for the evaluation results until the feedback is found.
  const evaluationResultsPromise = newFeedbackId
    ? pollForEvaluations(
        evaluation_name,
        datapoint_id,
        selectedRunIds,
        newFeedbackId,
      )
    : getEvaluationsForDatapoint(evaluation_name, datapoint_id, selectedRunIds);

  // Execute all promises concurrently
  const [
    selected_evaluation_run_infos,
    allowedEvaluationRunInfos,
    evaluationResults,
  ] = await Promise.all([
    selectedEvaluationRunInfosPromise,
    allowedEvaluationRunInfosPromise,
    evaluationResultsPromise,
  ]);

  const consolidatedEvaluationResults =
    consolidate_evaluation_results(evaluationResults);
  if (consolidatedEvaluationResults.length !== selectedRunIds.length) {
    // Find which evaluation run IDs are missing from the results
    const foundEvaluationRunIds = new Set(
      consolidatedEvaluationResults.map((result) => result.evaluation_run_id),
    );
    const missingEvaluationRunIds = selectedRunIds.filter(
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
    selected_evaluation_run_infos,
    allowedEvaluationRunInfos,
    selectedRunIds,
    newFeedbackId,
    newJudgeDemonstrationId,
  };
}

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();
  const _action = formData.get("_action");
  switch (_action) {
    case "addToDataset": {
      return handleAddToDatasetAction(formData);
    }
    case "addFeedback": {
      const response = await addEvaluationHumanFeedback(formData);
      const url = new URL(request.url);
      url.searchParams.delete("beforeFeedback");
      url.searchParams.delete("afterFeedback");
      url.searchParams.set(
        "newFeedbackId",
        response.feedbackResponse.feedback_id,
      );
      if (response.judgeDemonstrationResponse) {
        url.searchParams.set(
          "newJudgeDemonstrationId",
          response.judgeDemonstrationResponse.feedback_id,
        );
      } else {
        logger.warn("No judge demonstration response");
      }
      return redirect(url.toString());
    }
    case "renameDatapoint": {
      const datapoint_id = formData.get("datapoint_id") as string;
      const dataset_name = formData.get("dataset_name") as string;
      const newName = formData.get("newName") as string;

      // We need to get the datapoint to pass to renameDatapoint
      const datapoint = await getDatapoint({ dataset_name, datapoint_id });
      if (!datapoint) {
        return data(
          {
            success: false,
            error:
              "Datapoint not found; please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports",
          },
          { status: 404 },
        );
      }

      // A bit of a hack in the evaluation page, we don't have the function type in the datapoint, so we check if the datapoint contains an output schema (which indicates it's JSON).
      const functionType = "output_schema" in datapoint ? "json" : "chat";
      await renameDatapoint({
        functionType,
        datasetName: dataset_name,
        // TODO: convert to Rust-generated bindings
        datapoint,
        newName,
      });

      return data({ success: true });
    }
    case null:
      logger.error("No action provided");
      return null;
    default:
      logger.error(`Unknown action: ${_action}`);
      return null;
  }
}

export default function EvaluationDatapointPage({
  loaderData,
}: Route.ComponentProps) {
  const {
    consolidatedEvaluationResults,
    evaluation_name,
    datapoint_id,
    selected_evaluation_run_infos,
    allowedEvaluationRunInfos,
    selectedRunIds,
    newFeedbackId,
    newJudgeDemonstrationId,
  } = loaderData;
  const fetcher = useFetcher();
  const config = useConfig();
  const evaluation_config = config.evaluations[evaluation_name];
  if (!evaluation_config) {
    throw data(
      `Evaluation config not found for evaluation ${evaluation_name}`,
      { status: 404 },
    );
  }
  const outputsToDisplay = [
    ...(consolidatedEvaluationResults[0].reference_output !== null
      ? [
          {
            id: "Reference",
            output: consolidatedEvaluationResults[0].reference_output,
            metrics: [],
            variant_name: "Reference",
            inferenceId: null,
            episodeId: null,
          },
        ]
      : []),
    ...consolidatedEvaluationResults.map((result) => ({
      id: result.evaluation_run_id,
      inferenceId: result.inference_id,
      episodeId: result.episode_id,
      variant_name: result.variant_name,
      output: result.generated_output,
      metrics: result.metrics,
    })),
  ];
  const { toast } = useToast();
  useEffect(() => {
    if (newFeedbackId) {
      toast({
        title: "Feedback Added",
      });
    }
  }, [newFeedbackId, newJudgeDemonstrationId, toast]);

  const handleRenameDatapoint = async (newName: string) => {
    const formData = new FormData();
    formData.append("_action", "renameDatapoint");
    formData.append("datapoint_id", datapoint_id);
    formData.append(
      "dataset_name",
      consolidatedEvaluationResults[0].dataset_name,
    );
    formData.append("newName", newName);
    await fetcher.submit(formData, { method: "post", action: "." });
  };

  return (
    // Provider remains here
    <ColorAssignerProvider selectedRunIds={selectedRunIds}>
      <PageLayout>
        <PageHeader label="Datapoint" name={datapoint_id}>
          <BasicInfo
            evaluation_name={evaluation_name}
            evaluation_config={evaluation_config}
            dataset_name={consolidatedEvaluationResults[0].dataset_name}
            datapoint_name={consolidatedEvaluationResults[0].name}
            onRenameDatapoint={handleRenameDatapoint}
          />
          <EvalRunSelector
            evaluationName={evaluation_name}
            selectedRunIdInfos={selected_evaluation_run_infos}
            allowedRunInfos={allowedEvaluationRunInfos}
            // This must be passed so the component can filter by datapoint_id in search
          />
        </PageHeader>

        <SectionsGroup>
          <SectionLayout>
            <SectionHeader heading="Input" />
            <InputSnippet {...consolidatedEvaluationResults[0].input} />
          </SectionLayout>
          <OutputsSection
            outputsToDisplay={outputsToDisplay}
            evaluation_name={evaluation_name}
            evaluation_config={evaluation_config}
            datapointId={datapoint_id}
          />
        </SectionsGroup>
        <Toaster />
      </PageLayout>
    </ColorAssignerProvider>
  );
}

// Component to display metrics for a result
const MetricsDisplay = ({
  metrics,
  evaluation_name,
  evaluatorsConfig,
  datapointId,
  inferenceId,
  evalRunId,
  variantName,
}: {
  metrics: ConsolidatedMetric[];
  evaluation_name: string;
  evaluatorsConfig: Record<string, EvaluatorConfig | undefined>;
  datapointId: string;
  inferenceId: string | null;
  evalRunId: string;
  variantName: string;
}) => {
  return (
    <div className="pt-2">
      <div className="space-y-1">
        {metrics.map((metricObj) => {
          const evaluatorConfig = evaluatorsConfig[metricObj.evaluator_name];
          if (!evaluatorConfig) return null;

          return (
            <MetricRow
              // TODO(shuyangli): This may be the same across different rows.
              key={metricObj.evaluator_name}
              evaluation_name={evaluation_name}
              evaluatorName={metricObj.evaluator_name}
              metricValue={metricObj.metric_value}
              evaluatorConfig={evaluatorConfig}
              datapointId={datapointId}
              inferenceId={inferenceId}
              evaluatorInferenceId={metricObj.evaluator_inference_id}
              evalRunId={evalRunId}
              variantName={variantName}
              isHumanFeedback={metricObj.is_human_feedback}
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
  datapointId,
  inferenceId,
  evalRunId,
  evaluatorInferenceId,
  variantName,
  isHumanFeedback,
}: {
  evaluatorName: string;
  evaluation_name: string;
  metricValue: string;
  evaluatorConfig: EvaluatorConfig;
  datapointId: string;
  inferenceId: string | null;
  evaluatorInferenceId: string | null;
  evalRunId: string;
  variantName: string;
  isHumanFeedback: boolean;
}) => {
  const config = useConfig();
  const metric_name = getEvaluatorMetricName(evaluation_name, evaluatorName);
  const metricProperties = config.metrics[metric_name];
  if (!metricProperties) {
    return null;
  }
  if (inferenceId === null) {
    logger.warn(
      `Inference ID is null for metric ${metric_name} in datapoint ${datapointId}, this should not happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports`,
    );
  }
  const evaluationType = evaluatorConfig.type;
  return (
    <div className="group flex items-center gap-2">
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
        optimize={
          evaluatorConfig.type === "llm_judge"
            ? evaluatorConfig.optimize
            : "max"
        }
        cutoff={evaluatorConfig.cutoff ?? undefined}
        isHumanFeedback={isHumanFeedback}
        className="text-sm"
      />
      {inferenceId !== null && evaluationType === "llm_judge" && (
        <div className="flex gap-1 opacity-0 transition-opacity duration-200 group-hover:opacity-100">
          <EvaluationFeedbackEditor
            inferenceId={inferenceId}
            datapointId={datapointId}
            metricName={metric_name}
            originalValue={metricValue}
            evalRunId={evalRunId}
            evaluatorInferenceId={evaluatorInferenceId}
            variantName={variantName}
          />
          {evaluatorInferenceId && (
            <InferenceButton
              inferenceId={evaluatorInferenceId}
              tooltipText="View LLM judge inference"
            />
          )}
        </div>
      )}
    </div>
  );
};

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  logger.error(error);

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

type OutputsSectionProps = {
  outputsToDisplay: Array<{
    id: string;
    variant_name: string;
    output: ContentBlockChatOutput[] | JsonInferenceOutput;
    metrics: ConsolidatedMetric[];
    inferenceId: string | null;
    episodeId: string | null;
  }>;
  evaluation_name: string;
  evaluation_config: EvaluationConfig; // Use the specific config type
  datapointId: string;
};

function OutputsSection({
  outputsToDisplay,
  evaluation_name,
  evaluation_config,
  datapointId,
}: OutputsSectionProps) {
  const { getColor } = useColorAssigner();

  return (
    <SectionLayout>
      <SectionHeader heading="Output" />
      <div className="grid grid-flow-col grid-rows-[min-content_min-content_min-content] gap-x-4 gap-y-2 overflow-x-auto">
        {outputsToDisplay.map((result) => (
          <section className="contents" key={result.id}>
            <div className="row-start-1 flex flex-col gap-1">
              {result.id === "Reference" ? (
                <EvaluationRunBadge
                  runInfo={{
                    evaluation_run_id: "",
                    variant_name: result.variant_name,
                  }}
                  getColor={() => "bg-gray-100 text-gray-700"}
                />
              ) : (
                <>
                  <EvaluationRunBadge
                    runInfo={{
                      evaluation_run_id: result.id,
                      variant_name: result.variant_name,
                    }}
                    // Use the getColor obtained from the correct context
                    getColor={getColor}
                  />
                  {result.inferenceId && (
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                      <span>
                        Inference:{" "}
                        <Link
                          to={toInferenceUrl(result.inferenceId)}
                          className="text-blue-600 hover:text-blue-800 hover:underline"
                        >
                          {result.inferenceId}
                        </Link>
                      </span>
                      {result.inferenceId && result.episodeId && (
                        <AddToDatasetButton
                          inferenceId={result.inferenceId}
                          functionName={evaluation_config.function_name}
                          variantName={result.variant_name}
                          episodeId={result.episodeId}
                          hasDemonstration={false}
                          alwaysUseInherit={true}
                        />
                      )}
                    </div>
                  )}
                </>
              )}
            </div>

            <section className="row-start-2">
              <Output output={result.output} />
            </section>

            {result.id !== "Reference" &&
              result.metrics &&
              result.metrics.length > 0 && (
                <MetricsDisplay
                  evaluation_name={evaluation_name}
                  metrics={result.metrics}
                  evaluatorsConfig={evaluation_config.evaluators}
                  datapointId={datapointId}
                  inferenceId={result.inferenceId}
                  evalRunId={result.id}
                  variantName={result.variant_name}
                />
              )}
          </section>
        ))}
      </div>
    </SectionLayout>
  );
}
