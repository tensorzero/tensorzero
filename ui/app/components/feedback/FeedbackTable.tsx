import FeedbackValue from "~/components/feedback/FeedbackValue";
import { getMetricName } from "~/utils/clickhouse/helpers";
import type { FeedbackRow } from "~/types/tensorzero";
import FeedbackBadges from "~/components/feedback/FeedbackBadges";
import { TagsBadges } from "~/components/feedback/TagsBadges";
import { useConfig } from "~/context/config";
import { TableItemShortUuid, TableItemTime } from "~/components/ui/TableItems";
import { cn } from "~/utils/common";
import { Badge } from "../ui/badge";
import { Skeleton } from "../ui/skeleton";
import { useMemo } from "react";

export function FeedbackCardsSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 3 }).map((_, i) => (
        <div
          key={i}
          className="rounded-lg border border-border bg-bg-primary p-4"
        >
          <div className="mb-2 flex items-start justify-between gap-3">
            <div className="flex items-center gap-2">
              <Skeleton className="h-4 w-48" />
              <Skeleton className="h-5 w-14 rounded-full" />
              <Skeleton className="h-5 w-16 rounded-full" />
            </div>
            <Skeleton className="h-4 w-20 shrink-0" />
          </div>
          <div className="flex items-center gap-4">
            <Skeleton className="h-4 w-16" />
            <Skeleton className="h-4 w-20" />
          </div>
        </div>
      ))}
    </div>
  );
}

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
      feedback.some(
        (row) => row.type === "comment" && row.id !== latestCommentId,
      ) ||
      feedback.some(
        (row) =>
          row.type === "demonstration" && row.id !== latestDemonstrationId,
      ) ||
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

  if (feedback.length === 0) {
    return (
      <div className="text-fg-muted flex items-center justify-center rounded-lg border py-12 text-sm">
        No feedback found
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {feedback.map((item, index) => {
        const metricName = getMetricName(item);
        const metricConfig = metrics[metricName];
        const isLatestOfType =
          item.type === "comment"
            ? item.id === latestCommentId
            : item.type === "demonstration"
              ? item.id === latestDemonstrationId
              : latestFeedbackIdByMetric?.[item.metric_name] === item.id;

        return (
          <div
            key={`${item.id}-${index}`}
            data-testid={`feedback-card-${item.id}`}
            className={cn(
              "rounded-lg border border-border bg-bg-primary p-4",
              !isLatestOfType && anyOverwrites && "opacity-60",
            )}
          >
            <div className="mb-2 flex items-start justify-between gap-3">
              <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2">
                <span className="break-all font-mono text-sm">
                  {metricName}
                </span>
                {metricConfig && (
                  <FeedbackBadges metric={metricConfig} row={item} />
                )}
                {anyOverwrites &&
                  (isLatestOfType ? (
                    <Badge variant="secondary">Latest</Badge>
                  ) : (
                    <Badge
                      variant="outline"
                      className="border-red-400 text-red-400"
                    >
                      Overwritten
                    </Badge>
                  ))}
              </div>
              <div className="text-fg-tertiary shrink-0 text-xs">
                <TableItemTime timestamp={item.timestamp} />
              </div>
            </div>

            <div className="flex items-center gap-x-5 text-sm">
              <div className="flex items-center gap-1.5">
                <span className="text-fg-secondary text-sm">ID</span>
                <TableItemShortUuid id={item.id} />
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-fg-secondary text-sm">Value</span>
                <FeedbackValue feedback={item} metric={metricConfig} />
              </div>
            </div>

            {Object.keys(item.tags).length > 0 && (
              <div className="mt-3">
                <TagsBadges tags={item.tags} />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
