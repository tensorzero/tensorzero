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
import { Badge } from "~/components/ui/badge";

import {
  VariantSelector,
  getVariantColor,
  getLastUuidSegment,
} from "./VariantSelector";
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

// Component for variant label with color coding and run ID tooltip
const VariantLabel = ({
  runId,
  variantName,
  allRunIds,
}: {
  runId: string;
  variantName: string;
  allRunIds: EvaluationRunInfo[];
}) => {
  const colorClass = getVariantColor(runId, allRunIds);
  const runIdSegment = getLastUuidSegment(runId);

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            className={`${colorClass} flex cursor-help items-center gap-1.5 px-2 py-1`}
          >
            <span>{variantName}</span>
            <span className="border-l border-white/30 pl-1.5 text-xs opacity-80">
              {runIdSegment}
            </span>
          </Badge>
        </TooltipTrigger>
        <TooltipContent side="top" className="p-2">
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
  metric_names: string[];
}

export function EvaluationTable({
  available_eval_run_ids,
  eval_results,
  eval_statistics,
  metric_names,
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

  // Format statistics for display
  const formattedStats = useMemo(() => {
    const stats = new Map<string, Map<string, number>>();

    available_eval_run_ids.forEach((info) => {
      stats.set(info.eval_run_id, new Map());
    });

    eval_statistics.forEach((stat) => {
      const runStats = stats.get(stat.eval_run_id);
      if (runStats) {
        runStats.set(stat.metric_name, stat.mean_metric);
      }
    });

    return stats;
  }, [eval_statistics, available_eval_run_ids]);

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

  // Format statistic for display
  const formatStatValue = (value: number): React.ReactNode => {
    if (value >= 0 && value <= 1) {
      const percentage = Math.round(value * 100);
      return (
        <span className="whitespace-nowrap text-gray-700">{percentage}%</span>
      );
    }
    return (
      <span className="whitespace-nowrap text-gray-700">
        {value.toFixed(2)}
      </span>
    );
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
                    <TableHead className="text-center">Variant</TableHead>
                  )}
                  <TableHead className="py-2 text-center">
                    Generated Output
                  </TableHead>

                  {/* Dynamic metric columns */}
                  {metric_names.map((metric) => (
                    <TableHead key={metric} className="py-2 text-center">
                      <div>{metric.split("::").pop()}</div>
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>

              <TableBody>
                {/* Summary Row - Moved to top */}
                <TableRow className="bg-muted/50 font-medium">
                  <TableCell colSpan={2} className="text-left">
                    Summary ({uniqueDatapoints.length} inputs)
                  </TableCell>

                  {/* If showing variant column, add variant badges */}
                  {showVariantColumn ? (
                    <TableCell className="align-middle">
                      <div className="flex flex-col gap-2">
                        {selectedRunIds.map((runId) => (
                          <div
                            key={`summary-variant-${runId}`}
                            className="flex justify-center"
                          >
                            <VariantLabel
                              runId={runId}
                              variantName={
                                runIdToVariant.get(runId) || "Unknown"
                              }
                              allRunIds={available_eval_run_ids}
                            />
                          </div>
                        ))}
                      </div>
                    </TableCell>
                  ) : null}

                  {/* Empty cell for Generated Output column */}
                  <TableCell />

                  {/* Summary cells for each metric */}
                  {metric_names.map((metric) => (
                    <TableCell
                      key={metric}
                      className="text-center align-middle"
                    >
                      {selectedRunIds.map((runId) => {
                        const metricValue = formattedStats
                          .get(runId)
                          ?.get(metric);

                        return (
                          <div
                            key={`summary-${runId}-${metric}`}
                            className="flex justify-center py-1"
                          >
                            {metricValue !== undefined
                              ? formatStatValue(metricValue)
                              : "-"}
                          </div>
                        );
                      })}
                    </TableCell>
                  ))}
                </TableRow>

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

                          {/* Variant label - only if multiple variants are selected */}
                          {showVariantColumn && (
                            <TableCell className="text-center align-middle">
                              <VariantLabel
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
