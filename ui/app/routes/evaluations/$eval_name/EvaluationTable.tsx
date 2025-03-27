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

import { VariantSelector } from "./VariantSelector";
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
import { getOptimize, type EvaluatorConfig } from "~/utils/config/evals";
import { useColorAssigner } from "./ColorAssigner";
import { ColorAssignerProvider } from "./ColorAssigner";

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
            className="tooltip-scrollable max-h-[60vh] max-w-md overflow-auto shadow-sm"
            avoidCollisions={true}
          >
            <div className="flex h-full w-full items-center justify-center p-4">
              <pre className="w-full whitespace-pre-wrap text-xs">
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
            <div className="flex cursor-help items-center gap-1 overflow-hidden text-ellipsis whitespace-nowrap">
              <span className="font-mono text-sm">{inputSummary}</span>
            </div>
          </TooltipTrigger>
          <TooltipContent
            side="right"
            align="start"
            sideOffset={5}
            className="tooltip-scrollable max-h-[60vh] max-w-[500px] overflow-auto shadow-sm"
            avoidCollisions={true}
          >
            <div className="flex h-full w-full items-center justify-center p-4">
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
            className="tooltip-scrollable max-h-[60vh] max-w-[500px] overflow-auto shadow-sm"
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
  selected_eval_run_infos: EvaluationRunInfo[];
  eval_results: ParsedEvaluationResult[];
  eval_statistics: EvaluationStatistics[];
  evaluator_names: string[];
  eval_name: string;
  mostRecentEvalInferenceDates: Map<string, Date>;
}

export function EvaluationTable({
  selected_eval_run_infos,
  eval_results,
  eval_statistics,
  evaluator_names,
  eval_name,
  mostRecentEvalInferenceDates,
}: EvaluationTableProps) {
  const selectedRunIds = selected_eval_run_infos.map(
    (info) => info.eval_run_id,
  );

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

    // Sort datapoints by ID in descending order
    return Array.from(datapoints.values()).sort((a, b) =>
      b.id.localeCompare(a.id),
    );
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
    selected_eval_run_infos.forEach((info) => {
      map.set(info.eval_run_id, info.variant_name);
    });
    return map;
  }, [selected_eval_run_infos]);

  // Format metric value for display
  const formatMetricValue = (
    value: string,
    metricType: string,
    evaluatorConfig: EvaluatorConfig,
  ): React.ReactNode => {
    if (metricType === "boolean") {
      const boolValue = value === "true" || value === "1";
      const optimize = getOptimize(evaluatorConfig);
      const failed =
        (!boolValue && optimize === "max") || (boolValue && optimize === "min");
      const icon = failed ? (
        <X className="mr-1 h-3 w-3 flex-shrink-0" />
      ) : (
        <Check className="mr-1 h-3 w-3 flex-shrink-0" />
      );

      return (
        <span
          className={`flex items-center whitespace-nowrap ${failed ? "text-red-700" : "text-gray-700"}`}
        >
          {icon}
          {boolValue ? "True" : "False"}
        </span>
      );
    } else {
      // Try to parse as number
      const numValue = parseFloat(value);
      if (!isNaN(numValue)) {
        // Check if value fails the cutoff criteria
        const failsCutoff = isCutoffFailed(numValue, evaluatorConfig);
        return (
          <span
            className={`whitespace-nowrap ${failsCutoff ? "text-red-700" : "text-gray-700"}`}
          >
            {numValue}
          </span>
        );
      }

      // Default case: return as string
      return <span className="whitespace-nowrap">{value}</span>;
    }
  };
  const config = useConfig();

  return (
    <ColorAssignerProvider selectedRunIds={selectedRunIds}>
      <div>
        {/* Variant selector */}
        <VariantSelector
          evalName={eval_name}
          selectedRunIdInfos={selected_eval_run_infos}
          mostRecentEvalInferenceDates={mostRecentEvalInferenceDates}
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
                        {/* If there are multiple variants, we need to add a border to the last row only.
                            In the single-variant case the length should be 1 so every row will have a border.
                        */}
                        {filteredVariants.map(([runId, data], index) => (
                          <TableRow
                            key={`input-${datapoint.id}-variant-${runId}`}
                            className={
                              index !== filteredVariants.length - 1
                                ? "border-b-0"
                                : ""
                            }
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
                                eval_name,
                                evaluator_name,
                              );
                              const metricValue = data.metrics.get(metric_name);
                              const metricType =
                                config.metrics[metric_name].type;
                              const evaluatorConfig =
                                config.evals[eval_name].evaluators[
                                  evaluator_name
                                ];

                              return (
                                <TableCell
                                  key={metric_name}
                                  className="h-[52px] text-center align-middle"
                                >
                                  <div className="flex h-full items-center justify-center">
                                    {metricValue
                                      ? formatMetricValue(
                                          metricValue,
                                          metricType,
                                          evaluatorConfig,
                                        )
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
    </ColorAssignerProvider>
  );
}

const EvaluatorHeader = ({
  eval_name,
  evaluator_name,
  summaryStats,
}: {
  eval_name: string;
  evaluator_name: string;
  summaryStats: EvaluationStatistics[];
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
  const selectedRunIdsParam = searchParams.get("eval_run_ids") || "";
  const selectedRunIds = selectedRunIdsParam
    ? selectedRunIdsParam.split(",")
    : [];

  // Create a map of stats by run ID for easy lookup
  const statsByRunId = new Map(
    summaryStats.map((stat) => [stat.eval_run_id, stat]),
  );

  // Filter and sort stats according to the order in URL parameters
  const orderedStats = selectedRunIds
    .filter((runId) => statsByRunId.has(runId))
    .map((runId) => statsByRunId.get(runId)!);

  return (
    <div className="mt-2 flex flex-col items-center gap-1">
      {orderedStats.length > 0 && (
        <div className="mt-2 text-center text-xs text-muted-foreground">
          {orderedStats.map((stat) => {
            // Get the variant color for the circle using the run ID from the stat
            const variantColorClass = useColorAssigner().getColor(
              stat.eval_run_id,
              false,
            ); // Pass 'false' to get non-hover version

            return (
              <div
                key={stat.eval_run_id}
                className={`mt-1 flex items-center justify-center gap-1.5 ${
                  isCutoffFailed(stat.mean_metric, evaluatorConfig)
                    ? "text-red-700"
                    : ""
                }`}
              >
                <div
                  className={`h-2 w-2 rounded-full ${variantColorClass} flex-shrink-0`}
                ></div>
                <span>
                  {formatSummaryValue(stat.mean_metric, metricConfig)}
                  {stat.stderr_metric ? (
                    <>
                      {" "}
                      ± {formatSummaryValue(stat.stderr_metric, metricConfig)}
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

const formatSummaryValue = (value: number, metricConfig: MetricConfig) => {
  if (metricConfig.type === "boolean") {
    return `${Math.round(value * 100)}%`;
  } else if (metricConfig.type === "float") {
    return value.toFixed(2);
  }
  return value;
};

const isCutoffFailed = (
  value: number | boolean,
  evaluatorConfig: EvaluatorConfig,
) => {
  const numericValue = typeof value === "number" ? value : value ? 1 : 0;
  const optimize = getOptimize(evaluatorConfig);
  if (evaluatorConfig.cutoff === undefined) {
    return false;
  }
  if (optimize === "max") {
    return numericValue < evaluatorConfig.cutoff;
  } else {
    return numericValue > evaluatorConfig.cutoff;
  }
};
