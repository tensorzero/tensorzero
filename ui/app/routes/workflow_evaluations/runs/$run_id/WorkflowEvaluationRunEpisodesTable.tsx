import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import { formatDate } from "~/utils/date";
import { toEpisodeUrl } from "~/utils/urls";
import type {
  WorkflowEvaluationRunEpisodeWithFeedback,
  WorkflowEvaluationRunStatisticsByMetricName,
} from "~/utils/clickhouse/workflow_evaluations";
import { TooltipContent, TooltipTrigger } from "~/components/ui/tooltip";
import { Tooltip } from "~/components/ui/tooltip";
import { TooltipProvider } from "~/components/ui/tooltip";
import { useConfig } from "~/context/config";
import { formatMetricSummaryValue } from "~/utils/config/feedback";
import { TableItemShortUuid } from "~/components/ui/TableItems";
import KVChip from "~/components/ui/KVChip";
import MetricValue from "~/components/metric/MetricValue";

export default function WorkflowEvaluationRunEpisodesTable({
  episodes,
  statistics,
}: {
  episodes: WorkflowEvaluationRunEpisodeWithFeedback[];
  statistics: WorkflowEvaluationRunStatisticsByMetricName[];
}) {
  // Extract all unique metric names from all episodes
  const allMetricNames = new Set<string>();
  episodes.forEach((episode) => {
    episode.feedback_metric_names.forEach((metricName) => {
      allMetricNames.add(metricName);
    });
  });

  // Convert to sorted array for consistent column order
  const uniqueMetricNames = Array.from(allMetricNames).sort();
  const config = useConfig();

  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Task Name</TableHead>
            <TableHead>Episode ID</TableHead>
            <TableHead>Timestamp</TableHead>
            <TableHead>Tags</TableHead>
            {uniqueMetricNames.map((metricName) => (
              <TableHead key={metricName}>
                <MetricHeader metricName={metricName} statistics={statistics} />
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {episodes.length === 0 ? (
            <TableEmptyState message="No episodes found" />
          ) : (
            episodes.map((episode) => (
              <TableRow key={episode.episode_id}>
                <TableCell>
                  <code className="block overflow-hidden rounded font-mono text-ellipsis whitespace-nowrap transition-colors duration-300">
                    {episode.task_name ?? "-"}
                  </code>
                </TableCell>
                <TableCell className="max-w-[200px]">
                  <TableItemShortUuid
                    id={episode.episode_id}
                    link={toEpisodeUrl(episode.episode_id)}
                  />
                </TableCell>
                <TableCell>{formatDate(new Date(episode.timestamp))}</TableCell>
                <TableCell>
                  {(() => {
                    const filteredTags = Object.entries(episode.tags).filter(
                      ([k]) => !k.startsWith("tensorzero::"),
                    );
                    if (filteredTags.length === 0) {
                      return "-";
                    }
                    return (
                      <div className="flex flex-wrap gap-1">
                        {filteredTags.map(([k, v]) => (
                          <KVChip key={k} k={k} v={v} />
                        ))}
                      </div>
                    );
                  })()}
                </TableCell>
                {uniqueMetricNames.map((metricName) => {
                  const metricIndex =
                    episode.feedback_metric_names.indexOf(metricName);
                  const metricValue =
                    metricIndex !== -1
                      ? episode.feedback_values[metricIndex]
                      : null;
                  const metricConfig = config.metrics[metricName];
                  return (
                    <TableCell key={metricName} className="text-center">
                      <div className="flex justify-center">
                        {metricValue !== null && metricConfig ? (
                          <MetricValue
                            value={metricValue}
                            metricType={metricConfig.type}
                            optimize={metricConfig.optimize}
                            cutoff={
                              metricConfig.type === "boolean" ? 0.5 : undefined
                            }
                            isHumanFeedback={false}
                          />
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </div>
                    </TableCell>
                  );
                })}
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}

function MetricHeader({
  metricName,
  statistics,
}: {
  metricName: string;
  statistics: WorkflowEvaluationRunStatisticsByMetricName[];
}) {
  const metricStats = statistics.find(
    (stat) => stat.metric_name === metricName,
  );
  const config = useConfig();
  const metricConfig = config.metrics[metricName];
  return (
    <TooltipProvider delayDuration={300}>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex cursor-help flex-col items-center">
            <div className="font-mono">{metricName}</div>
            {metricStats && metricConfig && (
              <div className="text-muted-foreground mt-2 text-xs">
                <span>
                  {formatMetricSummaryValue(
                    metricStats.avg_metric,
                    metricConfig,
                  )}
                  {/* Display CI error if it's non-zero and available */}
                  {metricStats.ci_error != null &&
                  metricStats.ci_error !== 0 ? (
                    <>
                      {" "}
                      ±{" "}
                      {formatMetricSummaryValue(
                        metricStats.ci_error,
                        metricConfig,
                      )}
                    </>
                  ) : null}{" "}
                  (n={metricStats.count})
                </span>
              </div>
            )}
          </div>
        </TooltipTrigger>
        {metricConfig && (
          <TooltipContent side="top" className="p-3">
            <div className="space-y-1 text-left text-xs">
              <div>
                <span className="font-medium">Type:</span>
                <span className="ml-2 font-medium">{metricConfig.type}</span>
              </div>
              <div>
                {(metricConfig.type === "float" ||
                  metricConfig.type === "boolean") && (
                  <div>
                    <span className="font-medium">Optimize:</span>
                    <span className="ml-2 font-medium">
                      {metricConfig.optimize}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </TooltipContent>
        )}
      </Tooltip>
    </TooltipProvider>
  );
}
