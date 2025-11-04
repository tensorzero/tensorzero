import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import { toEpisodeUrl } from "~/utils/urls";
import type {
  WorkflowEvaluationRunEpisodeWithFeedback,
  WorkflowEvaluationRunStatisticsByMetricName,
} from "~/utils/clickhouse/workflow_evaluations";
import { TooltipContent, TooltipTrigger } from "~/components/ui/tooltip";
import { Tooltip } from "~/components/ui/tooltip";
import { useConfig } from "~/context/config";
import { formatMetricSummaryValue } from "~/utils/config/feedback";
import { TableItemShortUuid, TableItemTime } from "~/components/ui/TableItems";
import KVChip from "~/components/ui/KVChip";
import MetricValue from "~/components/metric/MetricValue";
import FeedbackValue from "~/components/feedback/FeedbackValue";
import type { FeedbackRow } from "~/types/tensorzero";

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
                <TableCell>
                  <TableItemTime timestamp={episode.timestamp} />
                </TableCell>
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
                  return (
                    <TableCell
                      key={metricName}
                      className="max-w-[200px] text-center"
                    >
                      {metricValue !== null ? (
                        <FeedbackMetricValue
                          metricName={metricName}
                          value={metricValue}
                          episodeId={episode.episode_id}
                        />
                      ) : (
                        <span className="text-gray-400">-</span>
                      )}
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

// Unified component that handles all feedback types (comments, floats, booleans)
// Converts the simple metric name/value format to FeedbackRow format
function FeedbackMetricValue({
  metricName,
  value,
  episodeId,
}: {
  metricName: string;
  value: string;
  episodeId: string;
}) {
  const config = useConfig();
  const metricConfig = config.metrics[metricName];

  // Handle comment type using FeedbackValue component
  if (metricName === "comment") {
    const feedback: FeedbackRow = {
      type: "comment",
      id: episodeId,
      target_id: episodeId,
      target_type: "episode",
      value: value,
      tags: {},
      timestamp: new Date().toISOString(),
    };
    return <FeedbackValue feedback={feedback} />;
  }

  // Handle float and boolean metrics with MetricValue
  if (metricConfig) {
    return (
      <MetricValue
        value={value}
        metricType={metricConfig.type}
        optimize={metricConfig.optimize}
        cutoff={metricConfig.type === "boolean" ? 0.5 : undefined}
        isHumanFeedback={false}
      />
    );
  }

  // Unknown metric type
  return <span className="text-gray-400">-</span>;
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

  // Handle comment type - just show the name without tooltip
  if (metricName === "comment") {
    return (
      <div className="flex flex-col items-center">
        <div className="font-mono">{metricName}</div>
      </div>
    );
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="flex cursor-help flex-col items-center">
          <div className="font-mono">{metricName}</div>
          {metricStats && metricConfig && (
            <div className="text-muted-foreground mt-2 text-xs">
              <span>
                {formatMetricSummaryValue(metricStats.avg_metric, metricConfig)}
                {/* Display CI error if it's non-zero and available */}
                {metricStats.ci_error != null && metricStats.ci_error !== 0 ? (
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
  );
}
