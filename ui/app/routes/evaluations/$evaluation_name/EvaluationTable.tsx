import { useMemo } from "react";
import React from "react";
import { useSearchParams, useNavigate } from "react-router";
import { Tooltip as RadixTooltip } from "radix-ui";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { Checkbox } from "~/components/ui/checkbox";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { toEvaluationDatapointUrl } from "~/utils/urls";

import { EvalRunSelector } from "~/components/evaluations/EvalRunSelector";
import type {
  EvaluationRunInfo,
  EvaluationStatistics,
  ParsedEvaluationResult,
} from "~/utils/clickhouse/evaluations";
import type { ZodDisplayInput } from "~/utils/clickhouse/common";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";

// Import the custom tooltip styles
import "./tooltip-styles.css";
import { useConfig } from "~/context/config";
import { getEvaluatorMetricName } from "~/utils/clickhouse/evaluations";
import {
  formatMetricSummaryValue,
  formatConfidenceInterval,
} from "~/utils/config/feedback";
import type {
  EvaluatorConfig,
  MetricConfig,
  JsonInferenceOutput,
  ContentBlockChatOutput,
} from "~/types/tensorzero";
import {
  useColorAssigner,
  ColorAssignerProvider,
} from "~/hooks/evaluations/ColorAssigner";
import MetricValue, { isCutoffFailed } from "~/components/metric/MetricValue";
import EvaluationFeedbackEditor from "~/components/evaluations/EvaluationFeedbackEditor";
import { InferenceButton } from "~/components/utils/InferenceButton";
import Input from "~/components/inference/Input";
import { logger } from "~/utils/logger";
import { TableItemText } from "~/components/ui/TableItems";

type TruncatedContentProps = (
  | {
      type: "text";
      content: string;
    }
  | {
      type: "input";
      content: ZodDisplayInput;
    }
  | {
      type: "output";
      content: JsonInferenceOutput | ContentBlockChatOutput[];
    }
) & {
  maxLength?: number;
};

const TruncatedContent = ({
  maxLength = 30,
  type,
  content,
}: TruncatedContentProps) => {
  const truncatedLabel =
    type === "text"
      ? content.length > maxLength
        ? content.slice(0, maxLength) + "..."
        : content
      : type === "input"
        ? getInputSummary(content)
        : getOutputSummary(content);

  return (
    <TruncatedContentTooltip truncatedLabel={truncatedLabel}>
      {type === "text" ? (
        <div className="flex h-full w-full items-center justify-center p-4">
          <pre className="w-full text-xs whitespace-pre-wrap">{content}</pre>
        </div>
      ) : type === "input" ? (
        <Input {...content} />
      ) : Array.isArray(content) ? (
        <ChatOutputElement output={content} />
      ) : (
        <JsonOutputElement output={content} />
      )}
    </TruncatedContentTooltip>
  );
};

const TruncatedContentTooltip: React.FC<
  React.PropsWithChildren<{
    truncatedLabel: string;
  }>
> = ({ children, truncatedLabel }) => (
  <Tooltip>
    <TooltipTrigger asChild>
      <div className="flex items-center gap-1 overflow-hidden text-ellipsis whitespace-nowrap">
        <span className="font-mono text-sm">{truncatedLabel}</span>
      </div>
    </TooltipTrigger>

    {/* TODO Reuse animations and such with existing Tooltip component. Other styling doesn't work as well here. */}
    <RadixTooltip.Content
      className="animate-in fade-in-0 zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 z-50 max-h-[60vh] max-w-[500px] overflow-auto rounded-lg text-xs shadow-lg"
      side="right"
      align="start"
      sideOffset={2}
      avoidCollisions={true}
    >
      {children}
    </RadixTooltip.Content>
  </Tooltip>
);

// Helper function to generate a summary of an Input object
function getInputSummary(input: ZodDisplayInput): string {
  if (!input || !input.messages || input.messages.length === 0) {
    return "Empty input";
  }

  // Get the first message's first text content
  const firstMessage = input.messages[0];
  if (!firstMessage.content || firstMessage.content.length === 0) {
    return `${firstMessage.role} message`;
  }

  const firstContent = firstMessage.content[0];

  if (firstContent.type === "text") {
    const text = firstContent.text;
    return text.length > 30 ? text.substring(0, 30) + "..." : text;
  }

  if (firstContent.type === "missing_function_text") {
    const text = firstContent.value;
    return text.length > 30 ? text.substring(0, 30) + "..." : text;
  }

  if (firstContent.type === "raw_text") {
    const text = firstContent.value;
    return text.length > 30 ? text.substring(0, 30) + "..." : text;
  }

  if (firstContent.type === "template") {
    const argsText = JSON.stringify(firstContent.arguments, null, 2);
    const summary = `${firstContent.name}: ${argsText}`;
    return summary.length > 30 ? summary.substring(0, 30) + "..." : summary;
  }

  return `${firstMessage.role} message (${firstContent.type})`;
}

// Helper function to generate a summary of an Output object
function getOutputSummary(
  output: JsonInferenceOutput | ContentBlockChatOutput[],
): string {
  if (Array.isArray(output)) {
    // It's ContentBlockChatOutput[]
    if (output.length === 0) return "Empty output";

    const firstBlock = output[0];
    if (firstBlock.type === "text") {
      return firstBlock.text.length > 30
        ? firstBlock.text.substring(0, 30) + "..."
        : firstBlock.text;
    }
    return `${firstBlock.type} output`;
  } else {
    // It's JsonInferenceOutput
    if (!output.raw) return "Empty output";

    return output.raw.length > 30
      ? output.raw.substring(0, 30) + "..."
      : output.raw;
  }
}

// Component for variant circle with color coding and tooltip
const VariantCircle = ({
  runId,
  variantName,
}: {
  runId: string;
  variantName: string;
}) => {
  const { getColor } = useColorAssigner();
  const colorClass = getColor(runId);

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className={`${colorClass} h-4 w-4 cursor-help rounded-full`} />
      </TooltipTrigger>
      <TooltipContent side="top" className="p-2">
        <p className="text-xs">
          Variant: <span className="font-mono text-xs">{variantName}</span>
        </p>
        <p className="text-xs">
          Run ID: <span className="font-mono text-xs">{runId}</span>
        </p>
      </TooltipContent>
    </Tooltip>
  );
};

interface EvaluationTableProps {
  selected_evaluation_run_infos: EvaluationRunInfo[];
  evaluation_results: ParsedEvaluationResult[];
  evaluation_statistics: EvaluationStatistics[];
  evaluator_names: string[];
  evaluation_name: string;
  selectedRows: Map<string, SelectedRowData>;
  setSelectedRows: React.Dispatch<
    React.SetStateAction<Map<string, SelectedRowData>>
  >;
}

interface MetricValueInfo {
  value: string;
  evaluator_inference_id: string | null;
  inference_id: string;
  is_human_feedback: boolean;
}

// Interface for tracking selected rows
export interface SelectedRowData {
  datapoint_id: string;
  evaluation_run_id: string;
  inference_id: string;
  variant_name: string;
  episode_id: string | null;
}

export function EvaluationTable({
  selected_evaluation_run_infos,
  evaluation_results,
  evaluation_statistics,
  evaluator_names,
  evaluation_name,
  selectedRows,
  setSelectedRows,
}: EvaluationTableProps) {
  const selectedRunIds = selected_evaluation_run_infos.map(
    (info) => info.evaluation_run_id,
  );
  const config = useConfig();
  const navigate = useNavigate();

  // Get all unique datapoints from the results
  const uniqueDatapoints = useMemo(() => {
    const datapoints = new Map<
      string,
      {
        id: string;
        name: string | null;
        input: ZodDisplayInput;
        reference_output: JsonInferenceOutput | ContentBlockChatOutput[] | null;
      }
    >();

    evaluation_results.forEach((result) => {
      if (!datapoints.has(result.datapoint_id)) {
        datapoints.set(result.datapoint_id, {
          id: result.datapoint_id,
          name: result.name,
          input: result.input,
          reference_output: result.reference_output,
        });
      }
    });

    // Sort datapoints by ID in descending order
    return Array.from(datapoints.values()).sort((a, b) =>
      b.id.localeCompare(a.id),
    );
  }, [evaluation_results]);

  // Organize results by datapoint and run ID
  const organizedResults = useMemo(() => {
    const organized = new Map<
      string, // datapoint id
      Map<
        string, // evaluation run id
        {
          generated_output: JsonInferenceOutput | ContentBlockChatOutput[];
          metrics: Map<string, MetricValueInfo>;
        }
      >
    >();

    // Initialize with empty maps for all datapoints
    uniqueDatapoints.forEach((datapoint) => {
      organized.set(datapoint.id, new Map());
    });

    // Fill in the results
    evaluation_results.forEach((result) => {
      if (!result.datapoint_id || !result.evaluation_run_id) return;

      const datapointMap = organized.get(result.datapoint_id);
      if (!datapointMap) return;

      if (!datapointMap.has(result.evaluation_run_id)) {
        datapointMap.set(result.evaluation_run_id, {
          generated_output: result.generated_output,
          metrics: new Map(),
        });
      }

      const runData = datapointMap.get(result.evaluation_run_id);
      if (runData && result.metric_name) {
        runData.metrics.set(result.metric_name, {
          value: result.metric_value,
          evaluator_inference_id: result.evaluator_inference_id,
          inference_id: result.inference_id,
          is_human_feedback: result.is_human_feedback,
        });
      }
    });

    return organized;
  }, [evaluation_results, uniqueDatapoints]);

  // Map run ID to variant name
  const runIdToVariant = useMemo(() => {
    const map = new Map<string, string>();
    selected_evaluation_run_infos.forEach((info) => {
      map.set(info.evaluation_run_id, info.variant_name);
    });
    return map;
  }, [selected_evaluation_run_infos]);

  // Build a map of row data for selection
  const rowDataMap = useMemo(() => {
    const map = new Map<string, SelectedRowData>();
    evaluation_results.forEach((result) => {
      const rowKey = `${result.datapoint_id}-${result.evaluation_run_id}`;
      if (result.inference_id && !map.has(rowKey)) {
        map.set(rowKey, {
          datapoint_id: result.datapoint_id,
          evaluation_run_id: result.evaluation_run_id,
          inference_id: result.inference_id,
          variant_name:
            runIdToVariant.get(result.evaluation_run_id) || "Unknown",
          episode_id: result.episode_id,
        });
      }
    });
    return map;
  }, [evaluation_results, runIdToVariant]);

  const handleRowSelect = (rowKey: string, rowData: SelectedRowData) => {
    setSelectedRows((prev) => {
      const newMap = new Map(prev);
      if (newMap.has(rowKey)) {
        newMap.delete(rowKey);
      } else {
        newMap.set(rowKey, rowData);
      }
      return newMap;
    });
  };

  const evaluation_config = config.evaluations[evaluation_name];
  if (!evaluation_config) {
    throw new Error(
      `Evaluation config not found for evaluation ${evaluation_name}`,
    );
  }

  return (
    <ColorAssignerProvider selectedRunIds={selectedRunIds}>
      <div>
        {/* Eval run selector */}
        <EvalRunSelector
          evaluationName={evaluation_name}
          selectedRunIdInfos={selected_evaluation_run_infos}
        />

        {selectedRunIds.length > 0 && (
          <div className="overflow-x-auto">
            <div className="min-w-max">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[40px] py-2 text-center align-top">
                      {/* Checkbox column */}
                    </TableHead>
                    <TableHead className="py-2 text-center align-top">
                      Name
                    </TableHead>
                    <TableHead className="py-2 text-center align-top">
                      Input
                    </TableHead>
                    <TableHead className="py-2 text-center align-top">
                      Reference Output
                    </TableHead>
                    {selectedRunIds.length > 1 && (
                      <TableHead className="py-2 text-center align-top">
                        {/* Empty header with minimal space */}
                      </TableHead>
                    )}
                    <TableHead className="py-2 text-center align-top">
                      Generated Output
                    </TableHead>
                    {/* Dynamic metric columns */}
                    {evaluator_names.map((evaluator_name) => {
                      // Get the metric name for this evaluator
                      const metric_name = getEvaluatorMetricName(
                        evaluation_name,
                        evaluator_name,
                      );

                      // Filter statistics for this specific metric
                      const filteredStats = evaluation_statistics.filter(
                        (stat) => stat.metric_name === metric_name,
                      );

                      return (
                        <TableHead
                          key={evaluator_name}
                          className="py-2 text-center"
                        >
                          <EvaluatorHeader
                            evaluation_name={evaluation_name}
                            evaluator_name={evaluator_name}
                            summaryStats={filteredStats}
                          />
                        </TableHead>
                      );
                    })}
                  </TableRow>
                </TableHeader>

                <TableBody>
                  {/* Map through datapoints and variants */}
                  {uniqueDatapoints.map((datapoint) => {
                    const variantData = organizedResults.get(datapoint.id);
                    if (!variantData) return null;

                    const filteredVariants = selectedRunIds
                      .map((runId) => [runId, variantData.get(runId)])
                      .filter(([, data]) => data !== undefined) as [
                      string,
                      {
                        generated_output:
                          | JsonInferenceOutput
                          | ContentBlockChatOutput[];
                        metrics: Map<string, MetricValueInfo>;
                      },
                    ][];

                    if (filteredVariants.length === 0) return null;

                    return (
                      <React.Fragment key={datapoint.id}>
                        {/* If there are multiple variants, we need to add a border to the last row only.
                            In the single-variant case the length should be 1 so every row will have a border.
                        */}
                        {filteredVariants.map(([runId, data], index) => {
                          const rowKey = `${datapoint.id}-${runId}`;
                          const isSelected = selectedRows.has(rowKey);
                          const rowData = rowDataMap.get(rowKey);

                          return (
                            <TableRow
                              key={`input-${datapoint.id}-variant-${runId}`}
                              className={
                                index !== filteredVariants.length - 1
                                  ? "cursor-pointer border-b-0"
                                  : "cursor-pointer"
                              }
                              onClick={() => {
                                const evaluation_run_ids = filteredVariants
                                  .map(([runId]) => runId)
                                  .join(",");
                                navigate(
                                  toEvaluationDatapointUrl(
                                    evaluation_name,
                                    datapoint.id,
                                    { evaluation_run_ids },
                                  ),
                                );
                              }}
                            >
                              {/* Checkbox cell */}
                              <TableCell
                                className="text-center align-middle"
                                onClick={(e) => e.stopPropagation()}
                              >
                                {rowData && (
                                  <Checkbox
                                    checked={isSelected}
                                    onCheckedChange={() =>
                                      handleRowSelect(rowKey, rowData)
                                    }
                                    aria-label={`Select row for ${datapoint.id}`}
                                  />
                                )}
                              </TableCell>

                              {/* Name cell - only for the first variant row */}
                              {index === 0 && (
                                <TableCell
                                  rowSpan={filteredVariants.length}
                                  className="max-w-[150px] text-center align-middle"
                                >
                                  <TableItemText text={datapoint.name} />
                                </TableCell>
                              )}

                              {/* Input cell - only for the first variant row */}
                              {index === 0 && (
                                <TableCell
                                  rowSpan={filteredVariants.length}
                                  className="max-w-[200px] align-middle"
                                >
                                  <TruncatedContent
                                    content={datapoint.input}
                                    type="input"
                                  />
                                </TableCell>
                              )}

                              {/* Reference Output cell - only for the first variant row */}
                              {index === 0 && (
                                <TableCell
                                  rowSpan={filteredVariants.length}
                                  className="max-w-[200px] text-center align-middle"
                                >
                                  {datapoint.reference_output ? (
                                    <TruncatedContent
                                      content={datapoint.reference_output}
                                      type="output"
                                    />
                                  ) : (
                                    "-"
                                  )}
                                </TableCell>
                              )}

                              {/* Variant circle - only if multiple variants are selected */}
                              {selectedRunIds.length > 1 && (
                                <TableCell className="text-center align-middle">
                                  <VariantCircle
                                    runId={runId}
                                    variantName={
                                      runIdToVariant.get(runId) || "Unknown"
                                    }
                                  />
                                </TableCell>
                              )}

                              {/* Generated output */}
                              <TableCell className="max-w-[200px] align-middle">
                                <TruncatedContent
                                  content={data.generated_output}
                                  type="output"
                                />
                              </TableCell>

                              {/* Metrics cells */}
                              {evaluator_names.map((evaluator_name) => {
                                const metric_name = getEvaluatorMetricName(
                                  evaluation_name,
                                  evaluator_name,
                                );
                                const metricValue =
                                  data.metrics.get(metric_name);
                                const metricType =
                                  config.metrics[metric_name]?.type;
                                const evaluatorConfig =
                                  config.evaluations[evaluation_name]
                                    ?.evaluators[evaluator_name];

                                return (
                                  <TableCell
                                    key={metric_name}
                                    className="h-[52px] text-center align-middle"
                                  >
                                    {/* Add group and relative positioning to the container */}
                                    <div className="group relative flex h-full items-center justify-center">
                                      {metricValue &&
                                      metricType &&
                                      evaluatorConfig ? (
                                        <>
                                          <MetricValue
                                            value={metricValue.value}
                                            metricType={metricType}
                                            isHumanFeedback={
                                              metricValue.is_human_feedback
                                            }
                                            optimize={
                                              evaluatorConfig.type ===
                                              "llm_judge"
                                                ? evaluatorConfig.optimize
                                                : "max"
                                            }
                                            cutoff={
                                              evaluatorConfig.cutoff ??
                                              undefined
                                            }
                                          />
                                          {/* Make feedback editor appear on hover */}
                                          {evaluatorConfig.type ===
                                            "llm_judge" && (
                                            <div
                                              className="absolute right-2 flex gap-1 opacity-0 transition-opacity duration-200 group-hover:opacity-100"
                                              // Stop click event propagation so the row navigation is not triggered
                                              onClick={(e) =>
                                                e.stopPropagation()
                                              }
                                            >
                                              <EvaluationFeedbackEditor
                                                inferenceId={
                                                  metricValue.inference_id
                                                }
                                                datapointId={datapoint.id}
                                                metricName={metric_name}
                                                originalValue={
                                                  metricValue.value
                                                }
                                                evalRunId={runId}
                                                evaluatorInferenceId={
                                                  metricValue.evaluator_inference_id
                                                }
                                                variantName={
                                                  runIdToVariant.get(runId) ||
                                                  "Unknown"
                                                }
                                              />
                                              {metricValue.evaluator_inference_id && (
                                                <InferenceButton
                                                  inferenceId={
                                                    metricValue.evaluator_inference_id
                                                  }
                                                  tooltipText="View LLM judge inference"
                                                />
                                              )}
                                            </div>
                                          )}
                                        </>
                                      ) : (
                                        "-"
                                      )}
                                    </div>
                                  </TableCell>
                                );
                              })}
                            </TableRow>
                          );
                        })}
                      </React.Fragment>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </div>
        )}
      </div>
    </ColorAssignerProvider>
  );
}

const EvaluatorHeader = ({
  evaluation_name,
  evaluator_name,
  summaryStats,
}: {
  evaluation_name: string;
  evaluator_name: string;
  summaryStats: EvaluationStatistics[];
}) => {
  const config = useConfig();
  const evaluationConfig = config.evaluations[evaluation_name];
  const evaluatorConfig = evaluationConfig?.evaluators[evaluator_name];
  if (!evaluatorConfig) {
    logger.warn(
      `Evaluator config not found for evaluation ${evaluation_name} and evaluator ${evaluator_name}`,
    );
    return null;
  }
  const metric_name = getEvaluatorMetricName(evaluation_name, evaluator_name);
  const metricProperties = config.metrics[metric_name];
  if (!metricProperties) {
    logger.warn(
      `Metric config not found for evaluation ${evaluation_name} and metric ${metric_name}`,
    );
    return null;
  }
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="cursor-help">
          <div className="font-mono">{evaluator_name}</div>
          <EvaluatorProperties
            metricConfig={metricProperties}
            summaryStats={summaryStats}
            evaluatorConfig={evaluatorConfig}
          />
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
              <span className="ml-2 font-medium">{evaluatorConfig.cutoff}</span>
            </div>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  );
};

const EvaluatorProperties = ({
  metricConfig,
  summaryStats,
  evaluatorConfig,
}: {
  metricConfig: MetricConfig;
  summaryStats: EvaluationStatistics[];
  evaluatorConfig: EvaluatorConfig;
}) => {
  const [searchParams] = useSearchParams();
  const selectedRunIdsParam = searchParams.get("evaluation_run_ids") || "";
  const selectedRunIds = selectedRunIdsParam
    ? selectedRunIdsParam.split(",")
    : [];

  // Create a map of stats by run ID for easy lookup
  const statsByRunId = new Map(
    summaryStats.map((stat) => [stat.evaluation_run_id, stat]),
  );

  // Filter and sort stats according to the order in URL parameters
  const orderedStats = selectedRunIds
    .filter((runId) => statsByRunId.has(runId))
    .map((runId) => statsByRunId.get(runId)!);

  const assigner = useColorAssigner();

  return (
    <div className="mt-2 flex flex-col items-center gap-1">
      {orderedStats.length > 0 && (
        <div className="text-muted-foreground mt-2 text-center text-xs">
          {orderedStats.map((stat) => {
            // Get the variant color for the circle using the run ID from the stat
            const variantColorClass = assigner.getColor(
              stat.evaluation_run_id,
              false,
            ); // Pass 'false' to get non-hover version
            const failed =
              evaluatorConfig.type === "llm_judge" && evaluatorConfig.cutoff
                ? isCutoffFailed(
                    stat.mean_metric,
                    evaluatorConfig.optimize,
                    evaluatorConfig.cutoff,
                  )
                : false;
            return (
              <div
                key={stat.evaluation_run_id}
                className={`mt-1 flex items-center justify-center gap-1.5 ${
                  failed ? "text-red-700" : ""
                }`}
              >
                <div
                  className={`h-2 w-2 rounded-full ${variantColorClass} shrink-0`}
                ></div>
                <span>
                  {formatMetricSummaryValue(stat.mean_metric, metricConfig)}
                  {stat.ci_lower != null && stat.ci_upper != null ? (
                    <>
                      {" "}
                      {formatConfidenceInterval(
                        stat.ci_lower,
                        stat.ci_upper,
                        metricConfig,
                      )}
                    </>
                  ) : null}{" "}
                  (n={stat.datapoint_count})
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
