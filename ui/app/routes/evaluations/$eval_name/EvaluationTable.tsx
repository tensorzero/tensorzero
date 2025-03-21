import { useMemo } from "react";
import React from "react";
import { useSearchParams } from "react-router";
import { Check, X } from "lucide-react";

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

import { VariantSelector, getVariantColor } from "./VariantSelector";
import type {
  EvaluationRunInfo,
  EvaluationStatistics,
  ParsedEvaluationResult,
} from "~/utils/clickhouse/evaluations";
import type { Input } from "~/utils/clickhouse/common";
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
import type { MetricConfig } from "~/utils/config/metric";

// Enhanced TruncatedText component that can handle complex structures
const TruncatedContent = ({
  content,
  maxLength = 30,
  type = "text",
}: {
  content: string | Input | JsonInferenceOutput | ContentBlockOutput[];
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
            <div className="flex cursor-help items-center gap-1 overflow-hidden text-ellipsis whitespace-nowrap">
              <span className="font-mono text-sm">{truncated}</span>
            </div>
          </TooltipTrigger>
          <TooltipContent
            side="right"
            align="start"
            sideOffset={5}
            className="tooltip-scrollable max-h-[60vh] max-w-md overflow-auto shadow-lg"
            avoidCollisions={true}
          >
            <pre className="whitespace-pre-wrap text-xs">{content}</pre>
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
            <div className="flex cursor-help items-center gap-1 overflow-hidden text-ellipsis whitespace-nowrap">
              <span className="font-mono text-sm">{inputSummary}</span>
            </div>
          </TooltipTrigger>
          <TooltipContent
            side="right"
            align="start"
            sideOffset={5}
            className="tooltip-scrollable max-h-[60vh] max-w-[500px] overflow-auto p-4 shadow-lg"
            avoidCollisions={true}
          >
            <div className="w-full origin-top-left scale-90 transform">
              <InputComponent input={content as Input} />
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
            <div className="flex cursor-help items-center gap-1 overflow-hidden text-ellipsis whitespace-nowrap">
              <span className="font-mono text-sm">{outputSummary}</span>
            </div>
          </TooltipTrigger>
          <TooltipContent
            side="right"
            align="start"
            sideOffset={5}
            className="tooltip-scrollable max-h-[60vh] max-w-[500px] overflow-auto p-4 shadow-lg"
            avoidCollisions={true}
          >
            <div className="w-full origin-top-left scale-90 transform">
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
  allRunIds,
}: {
  runId: string;
  variantName: string;
  allRunIds: EvaluationRunInfo[];
}) => {
  const colorClass = getVariantColor(runId, allRunIds);

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className={`${colorClass} h-4 w-4 cursor-help rounded-full`} />
        </TooltipTrigger>
        <TooltipContent side="top" className="p-2">
          <p className="text-xs">Variant: {variantName}</p>
          <p className="text-xs">Run ID: {runId}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

interface EvaluationTableProps {
  available_eval_run_ids: EvaluationRunInfo[];
  eval_results: ParsedEvaluationResult[];
  eval_statistics: EvaluationStatistics[];
  evaluator_names: string[];
  metric_names: string[];
  eval_name: string;
}

export function EvaluationTable({
  available_eval_run_ids,
  eval_results,
  eval_statistics,
  evaluator_names,
  metric_names,
  eval_name,
}: EvaluationTableProps) {
  const [searchParams] = useSearchParams();
  const selectedRunIdsParam = searchParams.get("eval_run_ids") || "";
  const selectedRunIds = selectedRunIdsParam
    ? selectedRunIdsParam.split(",")
    : [];

  // Determine if we should show the variant column
  const showVariantColumn = selectedRunIds.length > 1;

  // Get all unique datapoints from the results
  const uniqueDatapoints = useMemo(() => {
    const datapoints = new Map<
      string,
      {
        id: string;
        input: Input;
        reference_output: JsonInferenceOutput | ContentBlockOutput[];
      }
    >();

    eval_results.forEach((result) => {
      if (!datapoints.has(result.datapoint_id)) {
        datapoints.set(result.datapoint_id, {
          id: result.datapoint_id,
          input: result.input,
          reference_output: result.reference_output,
        });
      }
    });

    return Array.from(datapoints.values());
  }, [eval_results]);

  // Organize results by datapoint and run ID
  const organizedResults = useMemo(() => {
    const organized = new Map<
      string,
      Map<
        string,
        {
          generated_output: JsonInferenceOutput | ContentBlockOutput[];
          metrics: Map<string, string>;
        }
      >
    >();

    // Initialize with empty maps for all datapoints
    uniqueDatapoints.forEach((datapoint) => {
      organized.set(datapoint.id, new Map());
    });

    // Fill in the results
    eval_results.forEach((result) => {
      if (!result.datapoint_id || !result.eval_run_id) return;

      const datapointMap = organized.get(result.datapoint_id);
      if (!datapointMap) return;

      if (!datapointMap.has(result.eval_run_id)) {
        datapointMap.set(result.eval_run_id, {
          generated_output: result.generated_output,
          metrics: new Map(),
        });
      }

      const runData = datapointMap.get(result.eval_run_id);
      if (runData && result.metric_name) {
        runData.metrics.set(result.metric_name, result.metric_value);
      }
    });

    return organized;
  }, [eval_results, uniqueDatapoints]);

  // Map run ID to variant name
  const runIdToVariant = useMemo(() => {
    const map = new Map<string, string>();
    available_eval_run_ids.forEach((info) => {
      map.set(info.eval_run_id, info.variant_name);
    });
    return map;
  }, [available_eval_run_ids]);

  // Determine if metric is boolean or float based on its value
  const isMetricBoolean = (value: string): boolean => {
    return (
      value === "true" || value === "false" || value === "1" || value === "0"
    );
  };

  // Format metric value for display
  const formatMetricValue = (
    value: string,
    isBoolean = false,
  ): React.ReactNode => {
    if (isBoolean) {
      const boolValue = value === "true" || value === "1";
      const icon = boolValue ? (
        <Check className="mr-1 h-3 w-3 flex-shrink-0" />
      ) : (
        <X className="mr-1 h-3 w-3 flex-shrink-0" />
      );

      return (
        <span
          className={`flex items-center whitespace-nowrap ${boolValue ? "text-green-700" : "text-red-700"}`}
        >
          {icon}
          {boolValue ? "True" : "False"}
        </span>
      );
    } else {
      // Try to parse as number
      const numValue = parseFloat(value);
      if (!isNaN(numValue)) {
        // If it's between 0 and 1, display as percentage
        if (numValue >= 0 && numValue <= 1) {
          const percentage = Math.round(numValue * 100);
          return (
            <span className="whitespace-nowrap text-gray-700">
              {percentage}%
            </span>
          );
        }
        return (
          <span className="whitespace-nowrap text-gray-700">{numValue}</span>
        );
      }

      // Default case: return as string
      return <span className="whitespace-nowrap">{value}</span>;
    }
  };

  return (
    <div>
      {/* Variant selector */}
      <VariantSelector available_run_ids={available_eval_run_ids} />

      {selectedRunIds.length > 0 && (
        <div className="overflow-x-auto">
          <div className="min-w-max">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="py-2 text-center">Input</TableHead>
                  <TableHead className="py-2 text-center">
                    Reference Output
                  </TableHead>
                  {showVariantColumn && (
                    <TableHead className="py-2 text-center">
                      {/* Empty header with minimal space */}
                    </TableHead>
                  )}
                  <TableHead className="py-2 text-center">
                    Generated Output
                  </TableHead>
                  {/* Dynamic metric columns */}
                  {evaluator_names.map((evaluator_name) => {
                    // Get the metric name for this evaluator
                    const metric_name = getEvaluatorMetricName(
                      eval_name,
                      evaluator_name,
                    );

                    // Filter statistics for this specific metric
                    const filteredStats = eval_statistics.filter(
                      (stat) => stat.metric_name === metric_name,
                    );

                    return (
                      <TableHead
                        key={evaluator_name}
                        className="py-2 text-center"
                      >
                        <EvaluatorHeader
                          eval_name={eval_name}
                          evaluator_name={evaluator_name}
                          summaryStats={filteredStats}
                          evalRunIds={available_eval_run_ids}
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
                      metrics: Map<string, string>;
                    },
                  ][];

                  if (filteredVariants.length === 0) return null;

                  return (
                    <React.Fragment key={datapoint.id}>
                      {filteredVariants.map(([runId, data], index) => (
                        <TableRow
                          key={`input-${datapoint.id}-variant-${runId}`}
                        >
                          {/* Input cell - only for the first variant row */}
                          {index === 0 && (
                            <TableCell
                              rowSpan={filteredVariants.length}
                              className="max-w-[200px] align-top"
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
                              className="max-w-[200px] align-top"
                            >
                              <TruncatedContent
                                content={datapoint.reference_output}
                                type="output"
                              />
                            </TableCell>
                          )}

                          {/* Variant circle - only if multiple variants are selected */}
                          {showVariantColumn && (
                            <TableCell className="text-center align-middle">
                              <VariantCircle
                                runId={runId}
                                variantName={
                                  runIdToVariant.get(runId) || "Unknown"
                                }
                                allRunIds={available_eval_run_ids}
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
                          {metric_names.map((metric) => {
                            const metricValue = data.metrics.get(metric);
                            const isBoolean = metricValue
                              ? isMetricBoolean(metricValue)
                              : false;

                            return (
                              <TableCell
                                key={metric}
                                className="h-[52px] text-center align-middle"
                              >
                                <div className="flex h-full items-center justify-center">
                                  {metricValue
                                    ? formatMetricValue(metricValue, isBoolean)
                                    : "-"}
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
  );
}

const EvaluatorHeader = ({
  eval_name,
  evaluator_name,
  summaryStats,
  evalRunIds,
}: {
  eval_name: string;
  evaluator_name: string;
  summaryStats: EvaluationStatistics[];
  evalRunIds: EvaluationRunInfo[];
}) => {
  const config = useConfig();
  const evalConfig = config.evals[eval_name];
  const evaluatorConfig = evalConfig.evaluators[evaluator_name];
  const metric_name = getEvaluatorMetricName(eval_name, evaluator_name);
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
            <div>{evaluator_name}</div>
            <EvaluatorProperties
              metricConfig={metricProperties}
              summaryStats={summaryStats}
              evalRunIds={evalRunIds}
            />
          </div>
        </TooltipTrigger>
        <TooltipContent side="top" className="p-3">
          <div className="space-y-1 text-left text-xs">
            <div>
              <span className="font-medium">Type:</span>
              <span className="ml-2 font-mono">{metricProperties.type}</span>
            </div>
            <div>
              <span className="font-medium">Optimize:</span>
              <span className="ml-2 font-mono">
                {metricProperties.optimize}
              </span>
            </div>
            {evaluatorConfig.cutoff !== null && (
              <div>
                <span className="font-medium">Cutoff:</span>
                <span className="ml-2 font-mono">{evaluatorConfig.cutoff}</span>
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
  evalRunIds,
}: {
  metricConfig: MetricConfig;
  summaryStats: EvaluationStatistics[];
  evalRunIds: EvaluationRunInfo[];
}) => {
  console.log(evalRunIds);
  console.log(summaryStats);
  return (
    <div className="mt-2 flex flex-col items-center gap-1">
      {summaryStats && (
        <div className="mt-2 text-center text-xs text-muted-foreground">
          {summaryStats.map((stat, index) => {
            // Get the variant color for the circle
            const variantColorClass = getVariantColor(
              evalRunIds[index].eval_run_id,
              evalRunIds,
            );

            return (
              <div
                key={index}
                className="mt-1 flex items-center justify-center gap-1.5"
              >
                <div
                  className={`h-2 w-2 rounded-full ${variantColorClass} flex-shrink-0`}
                ></div>
                <span>
                  {formatSummaryValue(stat.mean_metric, metricConfig)} ± 0.05
                  (n=
                  {stat.datapoint_count})
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// Update the formatSummaryValue function to use percentages for boolean and decimals for float
const formatSummaryValue = (value: number, metricConfig: MetricConfig) => {
  if (metricConfig.type === "boolean") {
    return `${Math.round(value * 100)}%`;
  } else if (metricConfig.type === "float") {
    return value.toFixed(2);
  }
  return value;
};
