import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmptyState,
} from "~/components/ui/table";
import FeedbackValue from "~/components/feedback/FeedbackValue";
import { getMetricName } from "~/utils/clickhouse/helpers";
import type { FeedbackRow } from "~/types/tensorzero";
import FeedbackBadges from "~/components/feedback/FeedbackBadges";
import { TagsBadges } from "~/components/feedback/TagsBadges";
import { useConfig } from "~/context/config";
import { TableItemShortUuid, TableItemTime } from "~/components/ui/TableItems";
import { cn } from "~/utils/common";
import { Badge } from "../ui/badge";
import { useMemo } from "react";

export default function FeedbackTable({
  feedback,
  latestCommentId,
  latestDemonstrationId,
  latestFeedbackIdByMetric,
}: {
  feedback: FeedbackRow[];
  latestCommentId?: string;
  latestDemonstrationId?: string;
  latestFeedbackIdByMetric?: Record<string, string>;
}) {
  const config = useConfig();
  const metrics = config.metrics;

  const anyOverwrites = useMemo(() => {
    // Metric name => array of feedback IDs
    const metricToItem = feedback
      .filter((item) => "metric_name" in item)
      .reduce<Record<string, string[]>>(
        (metrics, { metric_name, id }) => ({
          ...metrics,
          [metric_name]: [...(metrics[metric_name] ?? []), id],
        }),
        {},
      );

    return (
      // Any comment that's not the latest
      feedback.some(
        (row) => row.type === "comment" && row.id !== latestCommentId,
      ) ||
      // Any demonstration that's not the latest
      feedback.some(
        (row) =>
          row.type === "demonstration" && row.id !== latestDemonstrationId,
      ) ||
      // Any metric where any feedback is not the latest for that metric
      Object.entries(metricToItem).some(([metric_name, ids]) => {
        return ids.some((id) => id !== latestFeedbackIdByMetric?.[metric_name]);
      })
    );
  }, [
    feedback,
    latestCommentId,
    latestDemonstrationId,
    latestFeedbackIdByMetric,
  ]);

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>ID</TableHead>
          <TableHead>Metric</TableHead>
          {anyOverwrites && <TableHead />}
          <TableHead>Value</TableHead>
          <TableHead>Tags</TableHead>
          <TableHead>Time</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {feedback.length === 0 ? (
          <TableEmptyState message="No feedback found" />
        ) : (
          feedback.map((item, index) => {
            const isLatestOfType =
              item.type === "comment"
                ? item.id === latestCommentId
                : item.type === "demonstration"
                  ? item.id === latestDemonstrationId
                  : latestFeedbackIdByMetric?.[item.metric_name] === item.id;

            return (
              <TableRow key={`${item.id}-${index}`}>
                <TableCell className="max-w-[200px]">
                  <TableItemShortUuid id={item.id} />
                </TableCell>

                <TableCell className="flex items-center gap-2">
                  <span className="font-mono">{getMetricName(item)}</span>
                  {metrics[getMetricName(item)] && (
                    <FeedbackBadges
                      metric={metrics[getMetricName(item)]!}
                      row={item}
                    />
                  )}
                </TableCell>

                {anyOverwrites && (
                  <TableCell>
                    {isLatestOfType ? (
                      <Badge variant="secondary">Latest</Badge>
                    ) : (
                      <Badge
                        variant="outline"
                        className="border-red-400 text-red-400"
                      >
                        Overwritten
                      </Badge>
                    )}
                  </TableCell>
                )}

                <TableCell
                  className={cn(
                    "max-w-[200px]",
                    !isLatestOfType && "opacity-50",
                  )}
                >
                  <FeedbackValue
                    feedback={item}
                    metric={metrics[getMetricName(item)]}
                  />
                </TableCell>

                <TableCell className={cn(!isLatestOfType && "opacity-50")}>
                  <TagsBadges tags={item.tags} />
                </TableCell>

                <TableCell>
                  <TableItemTime timestamp={item.timestamp} />
                </TableCell>
              </TableRow>
            );
          })
        )}
      </TableBody>
    </Table>
  );
}
