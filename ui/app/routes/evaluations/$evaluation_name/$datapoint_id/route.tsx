import {
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
  PageLayout,
  Breadcrumbs,
} from "~/components/layout/PageLayout";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

import {
  Await,
  data,
  Link,
  redirect,
  useFetcher,
  useLocation,
  type RouteHandle,
} from "react-router";
import { LayoutErrorBoundary } from "~/components/ui/error/LayoutErrorBoundary";
import { Suspense } from "react";
import { InputElement } from "~/components/input_output/InputElement";
import { EmptyMessage } from "~/components/input_output/ContentBlockElement";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";
import {
  consolidateEvaluationResults,
  getEvaluatorMetricName,
  type ConsolidatedMetric,
} from "~/utils/clickhouse/evaluations";
import MetricValue from "~/components/metric/MetricValue";
import { getMetricType } from "~/utils/config/evaluations";
import EvaluationRunBadge from "~/components/evaluations/EvaluationRunBadge";
import BasicInfo from "./EvaluationDatapointBasicInfo";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { EvalRunSelector } from "~/components/evaluations/EvalRunSelector";
import {
  ColorAssignerProvider,
  useColorAssigner,
} from "~/hooks/evaluations/ColorAssigner";
import { getConfig, getConfigForSnapshot } from "~/utils/config/index.server";
import type {
  EvaluatorConfig,
  InferenceEvaluationConfig,
  MetricConfig,
  JsonInferenceOutput,
  ContentBlockChatOutput,
} from "~/types/tensorzero";

import EvaluationFeedbackEditor from "~/components/evaluations/EvaluationFeedbackEditor";
import { InferenceButton } from "~/components/utils/InferenceButton";
import { addEvaluationHumanFeedback } from "~/utils/tensorzero.server";
import { handleAddToDatasetAction } from "~/utils/dataset.server";
import { renameDatapoint } from "~/routes/datasets/$dataset_name/datapoint/$id/datapointOperations.server";
import { useToast } from "~/hooks/use-toast";
import { useEffect } from "react";
import { AddToDatasetButton } from "~/components/dataset/AddToDatasetButton";
import { logger } from "~/utils/logger";
import { SectionAsyncErrorState } from "~/components/ui/error/ErrorContentPrimitives";
import { BasicInfoLayoutSkeleton } from "~/components/layout/BasicInfoLayout";
import { Skeleton } from "~/components/ui/skeleton";
import type { EvaluationRunInfo } from "~/utils/clickhouse/evaluations";
import type { ConsolidatedEvaluationResult } from "~/utils/clickhouse/evaluations";

export const handle: RouteHandle = {
  crumb: (match) => [
    "Datapoints",
    { label: match.params.datapoint_id!, isIdentifier: true },
  ],
};

interface RunInfoData {
  selected_evaluation_run_infos: EvaluationRunInfo[];
  allowedEvaluationRunInfos: EvaluationRunInfo[];
}

interface EvaluationResultsData {
  consolidatedEvaluationResults: ConsolidatedEvaluationResult[];
  datapoint_staled_at?: string;
}

async function fetchRunInfoData(
  datapoint_id: string,
  function_name: string,
  selectedRunIds: string[],
): Promise<RunInfoData> {
  const tensorZeroClient = getTensorZeroClient();

  const [selected_evaluation_run_infos, allowedEvaluationRunInfos] =
    await Promise.all([
      tensorZeroClient
        .getEvaluationRunInfos(selectedRunIds, function_name)
        .then((response) => response.run_infos),
      tensorZeroClient
        .getEvaluationRunInfosForDatapoint(datapoint_id, function_name)
        .then((response) => response.run_infos),
    ]);

  return {
    selected_evaluation_run_infos,
    allowedEvaluationRunInfos,
  };
}

async function fetchEvaluationResultsData(
  evaluation_name: string,
  datapoint_id: string,
  selectedRunIds: string[],
  newFeedbackId: string | null,
): Promise<EvaluationResultsData> {
  // If there is a freshly inserted feedback, ClickHouse may take some time to
  // update the evaluation results as it is eventually consistent.
  // In this case, we poll for the evaluation results until the feedback is found.
  const evaluationResults = newFeedbackId
    ? await pollForEvaluations(
        evaluation_name,
        datapoint_id,
        selectedRunIds,
        newFeedbackId,
      )
    : await getEvaluationsForDatapoint(
        evaluation_name,
        datapoint_id,
        selectedRunIds,
      );

  const consolidatedEvaluationResults =
    consolidateEvaluationResults(evaluationResults);
  if (consolidatedEvaluationResults.length !== selectedRunIds.length) {
    const foundEvaluationRunIds = new Set(
      consolidatedEvaluationResults.map((result) => result.evaluation_run_id),
    );
    const missingEvaluationRunIds = selectedRunIds.filter(
      (id) => !foundEvaluationRunIds.has(id),
    );

    throw new Error(
      `Evaluation run ID(s) not found: ${missingEvaluationRunIds.join(", ")}`,
    );
  }

  return {
    consolidatedEvaluationResults,
    datapoint_staled_at: consolidatedEvaluationResults[0].staled_at,
  };
}

export async function loader({ request, params }: Route.LoaderArgs) {
  const evaluation_name = params.evaluation_name;
  const datapoint_id = params.datapoint_id;
  const url = new URL(request.url);
  const searchParams = new URLSearchParams(url.search);
  const config = await getConfig();
  let evaluation_config = config.evaluations[evaluation_name];
  let effectiveConfig = config;

  if (!evaluation_config) {
    // Evaluation not in current config — try to find it from a historical snapshot
    const client = getTensorZeroClient();
    const runs = await client.listEvaluationRuns(100, 0);
    const matchingRun = runs.runs.find(
      (r) => r.evaluation_name === evaluation_name,
    );

    if (matchingRun?.snapshot_hash) {
      effectiveConfig = await getConfigForSnapshot(matchingRun.snapshot_hash);
      evaluation_config = effectiveConfig.evaluations[evaluation_name];
    }

    if (!evaluation_config) {
      throw data(
        `Evaluation config not found for evaluation ${evaluation_name}`,
        { status: 404 },
      );
    }
  }
  const function_name = evaluation_config.function_name;

  // Resolve function type from effective config
  // eslint-disable-next-line no-restricted-syntax
  const functionConfig = effectiveConfig.functions[function_name];
  const functionType = functionConfig?.type ?? ("chat" as const);

  // Build metrics config map for the relevant metrics
  const evaluator_names = Object.keys(evaluation_config.evaluators);
  const metricsConfig: Record<string, MetricConfig> = {};
  for (const evaluatorName of evaluator_names) {
    const metricName = getEvaluatorMetricName(evaluation_name, evaluatorName);
    const metricConfig = effectiveConfig.metrics[metricName];
    if (metricConfig) {
      metricsConfig[metricName] = metricConfig;
    }
  }

  const newFeedbackId = searchParams.get("newFeedbackId");
  const newJudgeDemonstrationId = searchParams.get("newJudgeDemonstrationId");

  const selected_evaluation_run_ids = searchParams.get("evaluation_run_ids");
  const selectedRunIds = selected_evaluation_run_ids
    ? selected_evaluation_run_ids.split(",")
    : [];
  if (selectedRunIds.length === 0) {
    return redirect(toEvaluationUrl(evaluation_name));
  }

  const tensorZeroClient = getTensorZeroClient();
  const tensorZeroDatapoint = await tensorZeroClient.getDatapoint(datapoint_id);
  if (!tensorZeroDatapoint) {
    throw data(`No datapoint found for ID \`${datapoint_id}\`.`, {
      status: 404,
    });
  }

  const runInfoData = fetchRunInfoData(
    datapoint_id,
    function_name,
    selectedRunIds,
  );
  const evaluationResultsData = fetchEvaluationResultsData(
    evaluation_name,
    datapoint_id,
    selectedRunIds,
    newFeedbackId,
  );

  return {
    evaluationConfig: evaluation_config,
    functionType,
    metricsConfig,
    runInfoData,
    evaluationResultsData,
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

      await renameDatapoint({
        datasetName: dataset_name,
        datapointId: datapoint_id,
        name: newName ?? null,
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

function EvalRunSelectorSkeleton() {
  return (
    <div className="flex flex-wrap gap-2">
      <Skeleton className="h-8 w-48" />
    </div>
  );
}

function InputSkeleton() {
  return <Skeleton className="h-32 w-full" />;
}

function OutputsSkeleton() {
  return (
    <div className="flex gap-4 overflow-x-auto">
      <div className="min-w-64 flex-1 space-y-2">
        <Skeleton className="h-8 w-32" />
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
      <div className="min-w-64 flex-1 space-y-2">
        <Skeleton className="h-8 w-32" />
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
    </div>
  );
}

function BasicInfoWithData({
  data,
  evaluation_name,
  evaluationConfig: evaluation_config,
  functionType,
  datapoint_id,
}: {
  data: EvaluationResultsData;
  evaluation_name: string;
  evaluationConfig: InferenceEvaluationConfig;
  functionType: "chat" | "json";
  datapoint_id: string;
}) {
  const { consolidatedEvaluationResults, datapoint_staled_at } = data;
  const fetcher = useFetcher();

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
    <BasicInfo
      evaluation_name={evaluation_name}
      evaluation_config={evaluation_config}
      functionType={functionType}
      dataset_name={consolidatedEvaluationResults[0].dataset_name}
      datapoint_id={datapoint_id}
      datapoint_name={consolidatedEvaluationResults[0].name}
      datapoint_staled_at={datapoint_staled_at}
      onRenameDatapoint={handleRenameDatapoint}
    />
  );
}

function MainContent({
  data,
  evaluation_name,
  evaluationConfig: evaluation_config,
  metricsConfig,
  datapoint_id,
}: {
  data: EvaluationResultsData;
  evaluation_name: string;
  evaluationConfig: InferenceEvaluationConfig;
  metricsConfig: Record<string, MetricConfig>;
  datapoint_id: string;
}) {
  const { consolidatedEvaluationResults } = data;

  const outputsToDisplay = [
    ...(consolidatedEvaluationResults[0].reference_output != null
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

  return (
    <SectionsGroup>
      <SectionLayout>
        <SectionHeader heading="Input" />
        {consolidatedEvaluationResults[0].input ? (
          <InputElement input={consolidatedEvaluationResults[0].input} />
        ) : (
          <EmptyMessage message="No input" />
        )}
      </SectionLayout>
      <OutputsSection
        outputsToDisplay={outputsToDisplay}
        evaluation_name={evaluation_name}
        evaluation_config={evaluation_config}
        metricsConfig={metricsConfig}
        datapointId={datapoint_id}
      />
    </SectionsGroup>
  );
}

function EvalRunSelectorWithData({
  data,
  evaluationName,
}: {
  data: RunInfoData;
  evaluationName: string;
}) {
  return (
    <EvalRunSelector
      evaluationName={evaluationName}
      selectedRunIdInfos={data.selected_evaluation_run_infos}
      allowedRunInfos={data.allowedEvaluationRunInfos}
    />
  );
}

export default function EvaluationDatapointPage({
  loaderData,
  params,
}: Route.ComponentProps) {
  const {
    evaluationConfig,
    functionType,
    metricsConfig,
    runInfoData,
    evaluationResultsData,
    selectedRunIds,
    newFeedbackId,
    newJudgeDemonstrationId,
  } = loaderData;
  const location = useLocation();
  const { toast } = useToast();

  // Show toast when feedback is successfully added (outside Suspense to avoid repeating)
  useEffect(() => {
    if (newFeedbackId) {
      const { dismiss } = toast.success({ title: "Feedback Added" });
      return () => dismiss({ immediate: true });
    }
    return;
  }, [newFeedbackId, newJudgeDemonstrationId, toast]);

  return (
    <ColorAssignerProvider selectedRunIds={selectedRunIds}>
      <PageLayout>
        <PageHeader
          eyebrow={
            <Breadcrumbs
              segments={[
                { label: "Evaluations", href: "/evaluations" },
                {
                  label: params.evaluation_name,
                  href: toEvaluationUrl(params.evaluation_name),
                  isIdentifier: true,
                },
                { label: "Results" },
              ]}
            />
          }
          name={params.datapoint_id}
        >
          <Suspense
            key={`${location.key}-basicinfo`}
            fallback={<BasicInfoLayoutSkeleton rows={5} />}
          >
            <Await
              resolve={evaluationResultsData}
              errorElement={<SectionAsyncErrorState />}
            >
              {(resultsData) => (
                <BasicInfoWithData
                  data={resultsData}
                  evaluation_name={params.evaluation_name}
                  evaluationConfig={evaluationConfig}
                  functionType={functionType}
                  datapoint_id={params.datapoint_id}
                />
              )}
            </Await>
          </Suspense>
        </PageHeader>

        <Suspense
          key={`${location.key}-selector`}
          fallback={<EvalRunSelectorSkeleton />}
        >
          <Await
            resolve={runInfoData}
            errorElement={<SectionAsyncErrorState />}
          >
            {(infoData) => (
              <EvalRunSelectorWithData
                data={infoData}
                evaluationName={params.evaluation_name}
              />
            )}
          </Await>
        </Suspense>

        {/* Main content - depends on evaluation results */}
        <Suspense
          key={`${location.key}-content`}
          fallback={
            <SectionsGroup>
              <SectionLayout>
                <SectionHeader heading="Input" />
                <InputSkeleton />
              </SectionLayout>
              <SectionLayout>
                <SectionHeader heading="Output" />
                <OutputsSkeleton />
              </SectionLayout>
            </SectionsGroup>
          }
        >
          <Await
            resolve={evaluationResultsData}
            errorElement={<SectionAsyncErrorState />}
          >
            {(resultsData) => (
              <MainContent
                data={resultsData}
                evaluation_name={params.evaluation_name}
                evaluationConfig={evaluationConfig}
                metricsConfig={metricsConfig}
                datapoint_id={params.datapoint_id}
              />
            )}
          </Await>
        </Suspense>
      </PageLayout>
    </ColorAssignerProvider>
  );
}

const MetricsDisplay = ({
  metrics,
  evaluation_name,
  evaluatorsConfig,
  metricsConfig,
  datapointId,
  inferenceId,
  evalRunId,
  variantName,
}: {
  metrics: ConsolidatedMetric[];
  evaluation_name: string;
  evaluatorsConfig: Record<string, EvaluatorConfig | undefined>;
  metricsConfig: Record<string, MetricConfig>;
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
              metricsConfig={metricsConfig}
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

const MetricRow = ({
  evaluatorName,
  evaluation_name,
  metricValue,
  evaluatorConfig,
  metricsConfig,
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
  metricsConfig: Record<string, MetricConfig>;
  datapointId: string;
  inferenceId: string | null;
  evaluatorInferenceId?: string;
  evalRunId: string;
  variantName: string;
  isHumanFeedback: boolean;
}) => {
  const metric_name = getEvaluatorMetricName(evaluation_name, evaluatorName);
  const metricProperties = metricsConfig[metric_name];
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
              <span className="ml-2 font-medium">{metricProperties.type}</span>
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
  return <LayoutErrorBoundary error={error} />;
}

type OutputsSectionProps = {
  outputsToDisplay: Array<{
    id: string;
    variant_name: string;
    output?: ContentBlockChatOutput[] | JsonInferenceOutput;
    metrics: ConsolidatedMetric[];
    inferenceId: string | null;
    episodeId: string | null;
  }>;
  evaluation_name: string;
  evaluation_config: InferenceEvaluationConfig;
  metricsConfig: Record<string, MetricConfig>;
  datapointId: string;
};

function OutputsSection({
  outputsToDisplay,
  evaluation_name,
  evaluation_config,
  metricsConfig,
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
              {result.output === undefined ? (
                <EmptyMessage message="No output" />
              ) : Array.isArray(result.output) ? (
                <ChatOutputElement output={result.output} />
              ) : (
                <JsonOutputElement output={result.output} />
              )}
            </section>

            {result.id !== "Reference" &&
              result.metrics &&
              result.metrics.length > 0 && (
                <MetricsDisplay
                  evaluation_name={evaluation_name}
                  metrics={result.metrics}
                  evaluatorsConfig={evaluation_config.evaluators}
                  metricsConfig={metricsConfig}
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
