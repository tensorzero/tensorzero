import { useMemo, useState } from "react";
import { ChevronRight } from "lucide-react";
import FeedbackValue from "~/components/feedback/FeedbackValue";
import {
  getMetricName,
  getDisplayMetricName,
} from "~/utils/clickhouse/helpers";
import type { FeedbackRow, MetricConfig } from "~/types/tensorzero";
import { useConfig } from "~/context/config";
import { TableItemShortUuid, TableItemTime } from "~/components/ui/TableItems";
import {
  Table,
  TableBody,
  TableCell,
  TableEmptyState,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { Skeleton } from "~/components/ui/skeleton";
import { cn } from "~/utils/common";

export function FeedbackTableHeaders() {
  return (
    <TableHeader>
      <TableRow>
        <TableHead>Metric</TableHead>
        <TableHead>Value</TableHead>
        <TableHead className="text-right">Time</TableHead>
      </TableRow>
    </TableHeader>
  );
}

export function FeedbackTableSkeleton() {
  return (
    <Table>
      <FeedbackTableHeaders />
      <TableBody>
        {Array.from({ length: 3 }).map((_, i) => (
          <TableRow key={i}>
            <TableCell>
              <Skeleton className="h-4 w-32" />
            </TableCell>
            <TableCell>
              <Skeleton className="h-4 w-20" />
            </TableCell>
            <TableCell className="text-right">
              <Skeleton className="ml-auto h-4 w-28" />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

interface MetricTooltipContentProps {
  rawMetricName: string | null;
  metricConfig: MetricConfig | undefined;
  tags: Record<string, string | undefined>;
}

function MetricTooltipContent({
  rawMetricName,
  metricConfig,
  tags,
}: MetricTooltipContentProps) {
  const userTags = Object.entries(tags).filter(
    (entry): entry is [string, string] =>
      !entry[0].startsWith("tensorzero::") && typeof entry[1] === "string",
  );

  if (!rawMetricName && !metricConfig && userTags.length === 0) {
    return null;
  }

  return (
    <div className="max-w-xs space-y-1.5 text-xs">
      {rawMetricName && (
        <div className="break-all font-mono text-white/70">{rawMetricName}</div>
      )}
      {metricConfig && (
        <div className="flex gap-1.5 text-white/90">
          <span>Type: {metricConfig.type}</span>
          {metricConfig.optimize && (
            <span>· Optimize: {metricConfig.optimize}</span>
          )}
          <span>· Level: {metricConfig.level}</span>
        </div>
      )}
      {userTags.length > 0 && (
        <div>
          <div className="mb-0.5 text-white/60">Tags</div>
          {userTags.map(([key, val]) => (
            <div key={key} className="font-mono text-white/90">
              {key}={val}
            </div>
          ))}
        </div>
      )}
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
  const [showOverwritten, setShowOverwritten] = useState(false);

  const { latestFeedback, overwrittenFeedback } = useMemo(() => {
    const latest: FeedbackRow[] = [];
    const overwritten: FeedbackRow[] = [];

    for (const item of feedback) {
      const isLatest =
        item.type === "comment"
          ? item.id === latestCommentId
          : item.type === "demonstration"
            ? item.id === latestDemonstrationId
            : latestFeedbackIdByMetric?.[item.metric_name] === item.id;

      if (isLatest) {
        latest.push(item);
      } else {
        overwritten.push(item);
      }
    }

    return { latestFeedback: latest, overwrittenFeedback: overwritten };
  }, [
    feedback,
    latestCommentId,
    latestDemonstrationId,
    latestFeedbackIdByMetric,
  ]);

  if (feedback.length === 0) {
    return (
      <Table>
        <FeedbackTableHeaders />
        <TableBody>
          <TableEmptyState message="No feedback found" />
        </TableBody>
      </Table>
    );
  }

  return (
    <Table>
      <FeedbackTableHeaders />
      <TableBody>
        {latestFeedback.map((item) => (
          <FeedbackRowItem key={item.id} item={item} metrics={metrics} />
        ))}
        {overwrittenFeedback.length > 0 && (
          <>
            <TableRow>
              <TableCell colSpan={3} className="py-1.5">
                <button
                  type="button"
                  aria-expanded={showOverwritten}
                  onClick={() => setShowOverwritten(!showOverwritten)}
                  className="text-fg-muted hover:text-fg-secondary flex cursor-pointer items-center gap-1 text-xs"
                >
                  <ChevronRight
                    className={cn(
                      "h-3 w-3 transition-transform",
                      showOverwritten && "rotate-90",
                    )}
                  />
                  {overwrittenFeedback.length} overwritten
                </button>
              </TableCell>
            </TableRow>
            {showOverwritten &&
              overwrittenFeedback.map((item) => (
                <FeedbackRowItem
                  key={item.id}
                  item={item}
                  metrics={metrics}
                  className="opacity-50"
                />
              ))}
          </>
        )}
      </TableBody>
    </Table>
  );
}

interface FeedbackRowItemProps {
  item: FeedbackRow;
  metrics: { [key in string]: MetricConfig };
  className?: string;
}

function FeedbackRowItem({ item, metrics, className }: FeedbackRowItemProps) {
  const rawMetricName = getMetricName(item);
  const metricConfig = metrics[rawMetricName];
  const displayMetric = getDisplayMetricName(item);

  const tooltipContent = (
    <MetricTooltipContent
      rawMetricName={displayMetric.raw}
      metricConfig={metricConfig}
      tags={item.tags}
    />
  );

  const hasTooltip =
    displayMetric.raw ||
    metricConfig ||
    Object.keys(item.tags).some((k) => !k.startsWith("tensorzero::"));

  return (
    <TableRow data-testid={`feedback-row-${item.id}`} className={className}>
      <TableCell>
        {hasTooltip ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="cursor-default font-mono text-sm underline decoration-dotted underline-offset-4">
                {displayMetric.display}
              </span>
            </TooltipTrigger>
            <TooltipContent sideOffset={5}>{tooltipContent}</TooltipContent>
          </Tooltip>
        ) : (
          <span className="font-mono text-sm">{displayMetric.display}</span>
        )}
      </TableCell>
      <TableCell>
        <FeedbackValue feedback={item} metric={metricConfig} />
      </TableCell>
      <TableCell className="text-right">
        <div className="flex items-center justify-end gap-2">
          <span className="text-fg-tertiary text-xs">
            <TableItemTime timestamp={item.timestamp} />
          </span>
          <span className="text-fg-muted text-xs">
            <TableItemShortUuid id={item.id} />
          </span>
        </div>
      </TableCell>
    </TableRow>
  );
}
