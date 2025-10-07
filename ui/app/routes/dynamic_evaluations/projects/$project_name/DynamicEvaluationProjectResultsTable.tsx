import React from "react";
import { useSearchParams } from "react-router";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { toEpisodeUrl } from "~/utils/urls";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

import type { DynamicEvaluationRun } from "~/utils/clickhouse/dynamic_evaluations";

import { useConfig } from "~/context/config";
import { formatMetricSummaryValue } from "~/utils/config/feedback";
import { useColorAssigner } from "~/hooks/evaluations/ColorAssigner";
import MetricValue from "~/components/metric/MetricValue";
import type {
  GroupedDynamicEvaluationRunEpisodeWithFeedback,
  DynamicEvaluationRunStatisticsByMetricName,
} from "~/utils/clickhouse/dynamic_evaluations";
import { TableItemShortUuid } from "~/components/ui/TableItems";
import { formatDate } from "~/utils/date";
import type { MetricConfig } from "tensorzero-node";

interface DynamicEvaluationProjectResultsTableProps {
  selected_run_infos: DynamicEvaluationRun[];
  evaluation_results: GroupedDynamicEvaluationRunEpisodeWithFeedback[][];
  evaluation_statistics: Record<
    string,
    DynamicEvaluationRunStatisticsByMetricName[]
  >;
}

export function DynamicEvaluationProjectResultsTable({
  selected_run_infos,
  evaluation_results,
  evaluation_statistics,
}: DynamicEvaluationProjectResultsTableProps) {
  const selectedRunIds = selected_run_infos.map((info) => info.id);
  // Extract all metrics from statistics
  // (we use statistics instead of results because we want to include metrics that are not present in this page)
  const allMetricNames = new Set<string>();
  Object.values(evaluation_statistics).forEach((stats) => {
    stats.forEach((stat) => {
      allMetricNames.add(stat.metric_name);
    });
  });
  const statisticsByMetricName = convertStatsByRunIdToStatsByMetricName(
    evaluation_statistics,
    Array.from(allMetricNames),
    selectedRunIds,
  );
  // Convert to sorted array for consistent column order
  const uniqueMetricNames = Array.from(allMetricNames).sort();
  const config = useConfig();
  return (
    <div>
      {selectedRunIds.length > 0 && (
        <div className="overflow-x-auto">
          <div className="min-w-max">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="py-2 text-center align-top">
                    Task Name
                  </TableHead>
                  {selectedRunIds.length > 1 && (
                    <TableHead className="py-2 text-center align-top">
                      {/* Empty header with minimal space */}
                    </TableHead>
                  )}
                  <TableHead className="py-2 text-center align-top">
                    Episode ID
                  </TableHead>
                  <TableHead className="py-2 text-center align-top">
                    Timestamp
                  </TableHead>
                  {/* Dynamic metric columns */}
                  {uniqueMetricNames.map((metric_name) => {
                    // Get the stats for this metric
                    const filteredStats =
                      statisticsByMetricName.get(metric_name);
                    if (!filteredStats) return null;

                    return (
                      <TableHead
                        key={metric_name}
                        className="py-2 text-center align-top"
                      >
                        <MetricHeader
                          metric_name={metric_name}
                          summaryStats={filteredStats}
                        />
                      </TableHead>
                    );
                  })}
                </TableRow>
              </TableHeader>

              <TableBody>
                {/* Map through datapoints and variants */}
                {evaluation_results.map((task_results) => {
                  if (task_results.length === 0) return null;
                  // Sort the results so they match the order of selected_run_ids
                  task_results.sort((a, b) => {
                    const indexA = selectedRunIds.indexOf(a.run_id);
                    const indexB = selectedRunIds.indexOf(b.run_id);
                    return indexA - indexB;
                  });

                  return (
                    <React.Fragment key={task_results[0].group_key}>
                      {/* If there are multiple variants, we need to add a border to the last row only.
                            In the single-variant case the length should be 1 so every row will have a border.
                        */}
                      {task_results.map((result, index) => (
                        <TableRow
                          key={`input-${result.group_key}`}
                          className={
                            index !== task_results.length - 1
                              ? "border-b-0"
                              : ""
                          }
                        >
                          {/* Task name cell - only for the first row in each group */}
                          {index === 0 && (
                            <TableCell
                              rowSpan={task_results.length}
                              className="max-w-[200px] overflow-hidden text-center align-middle text-ellipsis whitespace-nowrap"
                            >
                              {result.task_name ?? "-"}
                            </TableCell>
                          )}

                          {/* Variant circle - only if multiple variants are selected */}
                          {selectedRunIds.length > 1 && (
                            <TableCell className="align-middle">
                              <div className="flex h-full w-full items-center justify-center">
                                <EvaluationRunCircle runId={result.run_id} />
                              </div>
                            </TableCell>
                          )}

                          {/* Episode ID cell */}
                          <TableCell className="text-center align-middle">
                            <TableItemShortUuid
                              id={result.episode_id}
                              link={toEpisodeUrl(result.episode_id)}
                            />
                          </TableCell>

                          {/* Timestamp cell */}
                          <TableCell className="text-center align-middle">
                            {formatDate(new Date(result.timestamp))}
                          </TableCell>

                          {/* Metrics cells */}
                          {uniqueMetricNames.map((metric_name) => {
                            // Build a map for this row
                            const feedbackMap = new Map<string, string>();
                            result.feedback_metric_names.forEach(
                              (name, idx) => {
                                feedbackMap.set(
                                  name,
                                  result.feedback_values[idx],
                                );
                              },
                            );

                            const value = feedbackMap.get(metric_name);
                            const metricConfig = config.metrics[metric_name];

                            return (
                              <TableCell
                                key={metric_name}
                                className="h-[52px] text-center align-middle"
                              >
                                {/* Add group and relative positioning to the container */}
                                <div
                                  className={`group relative flex h-full items-center justify-center`}
                                >
                                  {value && metricConfig ? (
                                    <>
                                      <MetricValue
                                        value={value}
                                        metricType={metricConfig.type}
                                        optimize={metricConfig.optimize}
                                        isHumanFeedback={false}
                                      />
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
  );
}

const MetricHeader = ({
  metric_name,
  summaryStats,
}: {
  metric_name: string;
  summaryStats: DynamicEvaluationStatisticsByRunId[];
}) => {
  const config = useConfig();
  const metricProperties = config.metrics[metric_name];
  if (!metricProperties) {
    return null;
  }
  return (
    <TooltipProvider delayDuration={300}>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="cursor-help">
            <div className="font-mono">{metric_name}</div>
            <MetricProperties
              metricConfig={metricProperties}
              summaryStats={summaryStats}
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
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

const MetricProperties = ({
  metricConfig,
  summaryStats,
}: {
  metricConfig: MetricConfig;
  summaryStats: DynamicEvaluationStatisticsByRunId[];
}) => {
  const [searchParams] = useSearchParams();
  const selectedRunIdsParam = searchParams.get("run_ids") || "";
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
            return (
              <div
                key={stat.evaluation_run_id}
                className={`mt-1 flex items-center justify-center gap-1.5`}
              >
                <div
                  className={`h-2 w-2 rounded-full ${variantColorClass} shrink-0`}
                ></div>
                <span>
                  {formatMetricSummaryValue(stat.avg_metric, metricConfig)}
                  {stat.ci_error ? (
                    <>
                      {" "}
                      ± {formatMetricSummaryValue(stat.ci_error, metricConfig)}
                    </>
                  ) : null}{" "}
                  (n={stat.count})
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// Component for variant circle with color coding and tooltip
const EvaluationRunCircle = ({ runId }: { runId: string }) => {
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
            Run ID: <span className="font-mono text-xs">{runId}</span>
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

// Convert statistics from a map from run_id to a list of statistics with different metrics
// to a map from metric_name to a list of statistics with different run_ids,
// sorted in the order of selected_run_ids.
function convertStatsByRunIdToStatsByMetricName(
  evaluation_statistics: Record<
    string,
    DynamicEvaluationRunStatisticsByMetricName[]
  >,
  uniqueMetricNames: string[],
  selectedRunIds: string[],
) {
  const statisticsByMetricName = new Map<
    string,
    DynamicEvaluationStatisticsByRunId[]
  >();

  // Initialize the map with empty arrays for each metric name
  uniqueMetricNames.forEach((metricName) => {
    statisticsByMetricName.set(metricName, []);
  });

  // Populate the map with statistics for each metric
  selectedRunIds.forEach((runId) => {
    const runStats = evaluation_statistics[runId] || [];

    runStats.forEach((stat) => {
      // Create a new object with the evaluation_run_id property
      const statWithRunId = {
        ...stat,
        evaluation_run_id: runId,
      };

      const metricName = stat.metric_name;
      if (statisticsByMetricName.has(metricName)) {
        statisticsByMetricName
          .get(metricName)
          ?.push(statWithRunId as DynamicEvaluationStatisticsByRunId);
      }
    });
  });

  // Ensure statistics for each metric are sorted in the same order as selectedRunIds
  for (const [metricName, stats] of statisticsByMetricName.entries()) {
    statisticsByMetricName.set(
      metricName,
      stats.sort((a, b) => {
        const indexA = selectedRunIds.indexOf(a.evaluation_run_id);
        const indexB = selectedRunIds.indexOf(b.evaluation_run_id);
        return indexA - indexB;
      }),
    );
  }

  return statisticsByMetricName;
}

interface DynamicEvaluationStatisticsByRunId {
  evaluation_run_id: string;
  metric_name: string;
  count: number;
  avg_metric: number;
  stdev: number | null;
  ci_error: number | null;
}
