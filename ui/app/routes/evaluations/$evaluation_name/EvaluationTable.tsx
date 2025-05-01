import { useMemo } from "react";
import React from "react";
import { useSearchParams, useNavigate } from "react-router";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

import { EvalRunSelector } from "~/components/evaluations/EvalRunSelector";
import type {
  EvaluationRunInfo,
  EvaluationStatistics,
  ParsedEvaluationResult,
} from "~/utils/clickhouse/evaluations";
import type { Input, ResolvedInput } from "~/utils/clickhouse/common";
import type {
  JsonInferenceOutput,
  ContentBlockOutput,
} from "~/utils/clickhouse/common";
import InputComponent from "~/components/inference/Input";
import OutputComponent from "~/components/inference/Output";

// Import the custom tooltip styles
import "./tooltip-styles.css";
import { useConfig } from "~/context/config";
import { getEvaluatorMetricName } from "~/utils/clickhouse/evaluations";
import {
  formatMetricSummaryValue,
  type MetricConfig,
} from "~/utils/config/metric";
import { type EvaluatorConfig } from "~/utils/config/evaluations";
import {
  useColorAssigner,
  ColorAssignerProvider,
} from "~/hooks/evaluations/ColorAssigner";
import MetricValue, { isCutoffFailed } from "~/components/metric/MetricValue";
import EvaluationFeedbackEditor from "~/components/evaluations/EvaluationFeedbackEditor";

// Enhanced TruncatedText component that can handle complex structures
const TruncatedContent = ({
  content,
  maxLength = 30,
  type = "text",
}: {
  content: string | ResolvedInput | JsonInferenceOutput | ContentBlockOutput[];
  maxLength?: number;
  type?: "text" | "input" | "output";
}) => {
  // For simple strings, use the existing TruncatedText component
  if (typeof content === "string" && type === "text") {
    const truncated =
      content.length > maxLength
        ? content.slice(0, maxLength) + "..."
        : content;

    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-1 overflow-hidden text-ellipsis whitespace-nowrap">
              <span className="font-mono text-sm">{truncated}</span>
            </div>
          </TooltipTrigger>
          <TooltipContent
            side="right"
            align="start"
            sideOffset={5}
            className="tooltip-scrollable max-h-[60vh] max-w-md overflow-auto shadow-xs"
            avoidCollisions={true}
          >
            <div className="flex h-full w-full items-center justify-center p-4">
              <pre className="w-full text-xs whitespace-pre-wrap">
                {content}
              </pre>
            </div>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  // For Input type
  if (type === "input" && typeof content !== "string") {
    // For the truncated display, just show a brief summary
    const inputSummary = getInputSummary(content as Input);

    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-1 overflow-hidden text-ellipsis whitespace-nowrap">
              <span className="font-mono text-sm">{inputSummary}</span>
            </div>
          </TooltipTrigger>
          <TooltipContent
            side="right"
            align="start"
            sideOffset={5}
            className="tooltip-scrollable max-h-[60vh] max-w-[500px] overflow-auto shadow-xs"
            avoidCollisions={true}
          >
            <div className="flex h-full w-full items-center justify-center p-4">
              <InputComponent input={content as ResolvedInput} />
            </div>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  // For Output type
  if (type === "output" && typeof content !== "string") {
    // For the truncated display, just show a brief summary
    const outputSummary = getOutputSummary(
      content as JsonInferenceOutput | ContentBlockOutput[],
    );

    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-1 overflow-hidden text-ellipsis whitespace-nowrap">
              <span className="font-mono text-sm">{outputSummary}</span>
            </div>
          </TooltipTrigger>
          <TooltipContent
            side="right"
            align="start"
            sideOffset={5}
            className="tooltip-scrollable max-h-[60vh] max-w-[500px] overflow-auto shadow-xs"
            avoidCollisions={true}
          >
            <div className="flex h-full w-full items-center justify-center p-4">
              <OutputComponent
                output={content as JsonInferenceOutput | ContentBlockOutput[]}
              />
            </div>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  // Fallback for unknown types
  return <span>Unsupported content type</span>;
};

// Helper function to generate a summary of an Input object
function getInputSummary(input: Input): string {
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
    const text =
      typeof firstContent.value === "string"
        ? firstContent.value
        : JSON.stringify(firstContent.value);
    return text.length > 30 ? text.substring(0, 30) + "..." : text;
  }

  return `${firstMessage.role} message (${firstContent.type})`;
}

// Helper function to generate a summary of an Output object
function getOutputSummary(
  output: JsonInferenceOutput | ContentBlockOutput[],
): string {
  if (Array.isArray(output)) {
    // It's ContentBlockOutput[]
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
    <TooltipProvider>
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
    </TooltipProvider>
  );
};

interface EvaluationTableProps {
  selected_evaluation_run_infos: EvaluationRunInfo[];
  evaluation_results: ParsedEvaluationResult[];
  evaluation_statistics: EvaluationStatistics[];
  evaluator_names: string[];
  evaluation_name: string;
}

interface MetricValueInfo {
  value: string;
  evaluator_inference_id: string | null;
  inference_id: string;
}

export function EvaluationTable({
  selected_evaluation_run_infos,
  evaluation_results,
  evaluation_statistics,
  evaluator_names,
  evaluation_name,
}: EvaluationTableProps) {
  const selectedRunIds = selected_evaluation_run_infos.map(
    (info) => info.evaluation_run_id,
  );

  // Get all unique datapoints from the results
  const uniqueDatapoints = useMemo(() => {
    const datapoints = new Map<
      string,
      {
        id: string;
        input: ResolvedInput;
        reference_output: JsonInferenceOutput | ContentBlockOutput[];
      }
    >();

    evaluation_results.forEach((result) => {
      if (!datapoints.has(result.datapoint_id)) {
        datapoints.set(result.datapoint_id, {
          id: result.datapoint_id,
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
          generated_output: JsonInferenceOutput | ContentBlockOutput[];
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

  const config = useConfig();
  const navigate = useNavigate();
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
                          | ContentBlockOutput[];
                        metrics: Map<string, MetricValueInfo>;
                      },
                    ][];

                    if (filteredVariants.length === 0) return null;

                    return (
                      <React.Fragment key={datapoint.id}>
                        {/* If there are multiple variants, we need to add a border to the last row only.
                            In the single-variant case the length should be 1 so every row will have a border.
                        */}
                        {filteredVariants.map(([runId, data], index) => (
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
                                `/evaluations/${evaluation_name}/${datapoint.id}?evaluation_run_ids=${evaluation_run_ids}`,
                              );
                            }}
                          >
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
                                className="max-w-[200px] align-middle"
                              >
                                <TruncatedContent
                                  content={datapoint.reference_output}
                                  type="output"
                                />
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
                              const metricValue = data.metrics.get(metric_name);
                              const metricType =
                                config.metrics[metric_name].type;
                              const evaluatorConfig =
                                config.evaluations[evaluation_name].evaluators[
                                  evaluator_name
                                ];
                              const evaluationType = evaluatorConfig.type;

                              return (
                                <TableCell
                                  key={metric_name}
                                  className="h-[52px] text-center align-middle"
                                >
                                  {/* Add group and relative positioning to the container */}
                                  <div
                                    className={`group relative flex h-full items-center justify-center ${metricValue && evaluationType === "llm_judge" ? "pl-10" : ""}`}
                                  >
                                    {metricValue ? (
                                      <>
                                        <MetricValue
                                          value={metricValue.value}
                                          metricType={metricType}
                                          optimize={
                                            evaluatorConfig.type === "llm_judge"
                                              ? evaluatorConfig.optimize
                                              : "max"
                                          }
                                          cutoff={evaluatorConfig.cutoff}
                                        />
                                        {/* Make feedback editor appear on hover */}
                                        {evaluationType === "llm_judge" && (
                                          <div
                                            className="ml-2 opacity-0 transition-opacity duration-200 group-hover:opacity-100"
                                            // Stop click event propagation so the row navigation is not triggered
                                            onClick={(e) => e.stopPropagation()}
                                          >
                                            <EvaluationFeedbackEditor
                                              inferenceId={
                                                metricValue.inference_id
                                              }
                                              datapointId={datapoint.id}
                                              metricName={metric_name}
                                              originalValue={metricValue.value}
                                              evalRunId={runId}
                                              evaluatorInferenceId={
                                                metricValue.evaluator_inference_id
                                              }
                                              variantName={
                                                runIdToVariant.get(runId) ||
                                                "Unknown"
                                              }
                                            />
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
                        ))}
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
  const EvaluationConfig = config.evaluations[evaluation_name];
  const evaluatorConfig = EvaluationConfig.evaluators[evaluator_name];
  const metric_name = getEvaluatorMetricName(evaluation_name, evaluator_name);
  const metricProperties = config.metrics[metric_name];
  if (
    metricProperties.type === "comment" ||
    metricProperties.type === "demonstration"
  ) {
    return null;
  }
  return (
    <TooltipProvider delayDuration={300}>
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
                <span className="ml-2 font-medium">
                  {evaluatorConfig.cutoff}
                </span>
              </div>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
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
                  {stat.stderr_metric ? (
                    <>
                      {" "}
                      ±{" "}
                      {formatMetricSummaryValue(
                        stat.stderr_metric,
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
