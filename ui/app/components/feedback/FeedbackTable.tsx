import { useState, useMemo, useRef, useCallback } from "react";
import FeedbackValue from "~/components/feedback/FeedbackValue";
import {
  getMetricName,
  getDisplayMetricName,
} from "~/utils/clickhouse/helpers";
import type {
  FeedbackRow,
  FeedbackBounds,
  MetricConfig,
} from "~/types/tensorzero";
import { useConfig } from "~/context/config";
import { getFeedbackConfig } from "~/utils/config/feedback";
import type { FeedbackConfig } from "~/utils/config/feedback";
import { TableItemTime, TableItemShortUuid } from "~/components/ui/TableItems";
import { ChevronRight } from "lucide-react";
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
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import { Skeleton } from "~/components/ui/skeleton";

function formatTagKey(key: string): string {
  if (key.startsWith("tensorzero::")) {
    return key.slice("tensorzero::".length);
  }
  return key;
}

export function FeedbackTableHeaders() {
  return (
    <TableHeader>
      <TableRow>
        <TableHead className="w-[20%]">Metric</TableHead>
        <TableHead className="w-[36%]">Value</TableHead>
        <TableHead className="w-[8%]">Tags</TableHead>
        <TableHead className="w-[18%]">Config</TableHead>
        <TableHead className="w-[18%]">Time</TableHead>
      </TableRow>
    </TableHeader>
  );
}

export function FeedbackTableSkeleton() {
  return (
    <Table className="table-fixed">
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
            <TableCell>
              <Skeleton className="h-4 w-16" />
            </TableCell>
            <TableCell>
              <Skeleton className="h-4 w-24" />
            </TableCell>
            <TableCell>
              <Skeleton className="h-4 w-28" />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

interface MetricTooltipContentProps {
  id: string;
  rawMetricName: string | null;
}

function MetricTooltipContent({
  id,
  rawMetricName,
}: MetricTooltipContentProps) {
  return (
    <div className="max-w-xs space-y-2 text-xs">
      {rawMetricName && (
        <div className="break-all font-mono text-xs text-white/70">
          {rawMetricName}
        </div>
      )}
      <div className="break-all font-mono text-xs text-white/40">ID: {id}</div>
    </div>
  );
}

interface FeedbackTableProps {
  feedback: FeedbackRow[];
  feedbackBounds?: FeedbackBounds;
  latestByMetric?: Record<string, string>;
}

export default function FeedbackTable({
  feedback,
  feedbackBounds,
  latestByMetric,
}: FeedbackTableProps) {
  const config = useConfig();
  const [showOverwritten, setShowOverwritten] = useState(false);

  const { latest, overwritten } = useMemo(() => {
    if (!feedbackBounds || !latestByMetric) {
      return { latest: feedback, overwritten: [] };
    }
    const latestList = feedback.filter((item) => {
      if (item.type === "comment")
        return item.id === feedbackBounds.by_type.comment.last_id;
      if (item.type === "demonstration")
        return item.id === feedbackBounds.by_type.demonstration.last_id;
      return latestByMetric[item.metric_name] === item.id;
    });
    const latestIds = new Set(latestList.map((f) => f.id));
    return {
      latest: latestList,
      overwritten: feedback.filter((f) => !latestIds.has(f.id)),
    };
  }, [feedback, feedbackBounds, latestByMetric]);

  if (feedback.length === 0) {
    return (
      <Table className="table-fixed">
        <FeedbackTableHeaders />
        <TableBody>
          <TableEmptyState message="No feedback found" />
        </TableBody>
      </Table>
    );
  }

  return (
    <Table className="table-fixed">
      <FeedbackTableHeaders />
      <TableBody>
        {latest.map((item) => (
          <FeedbackRowItem
            key={item.id}
            item={item}
            metricConfig={config.metrics[getMetricName(item)]}
            feedbackConfig={getFeedbackConfig(getMetricName(item), config)}
          />
        ))}
        {overwritten.length > 0 && (
          <>
            <TableRow>
              <TableCell colSpan={5} className="py-1">
                <button
                  type="button"
                  className="text-fg-muted hover:text-fg-secondary flex items-center gap-1 text-xs transition-colors"
                  onClick={() => setShowOverwritten(!showOverwritten)}
                >
                  <ChevronRight
                    className={`h-3 w-3 transition-transform ${showOverwritten ? "rotate-90" : ""}`}
                  />
                  {overwritten.length} overwritten
                </button>
              </TableCell>
            </TableRow>
            {showOverwritten &&
              overwritten.map((item) => (
                <FeedbackRowItem
                  key={item.id}
                  item={item}
                  metricConfig={config.metrics[getMetricName(item)]}
                  feedbackConfig={getFeedbackConfig(
                    getMetricName(item),
                    config,
                  )}
                  isOverwritten
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
  metricConfig: MetricConfig | undefined;
  feedbackConfig: FeedbackConfig | undefined;
  isOverwritten?: boolean;
}

function FeedbackRowItem({
  item,
  metricConfig,
  feedbackConfig,
  isOverwritten,
}: FeedbackRowItemProps) {
  const displayMetric = getDisplayMetricName(item);

  const allTags = Object.entries(item.tags).filter(
    (entry): entry is [string, string] => typeof entry[1] === "string",
  );

  return (
    <TableRow
      data-testid={`feedback-row-${item.id}`}
      className={isOverwritten ? "opacity-50" : undefined}
    >
      <TableCell>
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="cursor-default font-mono text-sm underline decoration-dotted decoration-border underline-offset-4">
              {displayMetric.display}
            </span>
          </TooltipTrigger>
          <TooltipContent sideOffset={5}>
            <MetricTooltipContent
              id={item.id}
              rawMetricName={displayMetric.raw}
            />
          </TooltipContent>
        </Tooltip>
      </TableCell>
      <TableCell className="max-w-0 truncate">
        <FeedbackValue feedback={item} metric={metricConfig} />
      </TableCell>
      <TableCell>
        {allTags.length > 0 ? (
          <TagsPopover tags={allTags} />
        ) : (
          <span className="text-fg-muted text-xs">—</span>
        )}
      </TableCell>
      <TableCell>
        <MetricConfigInfo feedbackConfig={feedbackConfig} item={item} />
      </TableCell>
      <TableCell>
        <span className="text-fg-tertiary flex items-center gap-1.5 text-xs">
          <TableItemTime timestamp={item.timestamp} />
          <span className="text-fg-muted">·</span>
          <TableItemShortUuid id={item.id} />
        </span>
      </TableCell>
    </TableRow>
  );
}

interface MetricConfigInfoProps {
  feedbackConfig: FeedbackConfig | undefined;
  item: FeedbackRow;
}

function MetricConfigInfo({ feedbackConfig, item }: MetricConfigInfoProps) {
  if (!feedbackConfig) {
    return <span className="text-fg-muted text-xs">—</span>;
  }

  const parts: string[] = [];

  if (feedbackConfig.type === "comment") {
    parts.push("comment");
    if (item.type === "comment" && item.target_type) {
      parts.push(item.target_type);
    }
  } else if (feedbackConfig.type === "demonstration") {
    parts.push("demonstration");
    parts.push("inference");
  } else {
    parts.push(feedbackConfig.type);
    parts.push(feedbackConfig.optimize);
    parts.push(feedbackConfig.level);
  }

  return (
    <span className="text-fg-tertiary text-xs">
      {parts.map((part, i) => (
        <span key={i}>
          {i > 0 && (
            <span className="text-fg-muted mx-1" aria-hidden>
              ·
            </span>
          )}
          {part}
        </span>
      ))}
    </span>
  );
}

interface TagsPopoverProps {
  tags: [string, string][];
}

function TagsPopover({ tags }: TagsPopoverProps) {
  const [open, setOpen] = useState(false);
  const closeTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const scheduleClose = useCallback(() => {
    closeTimeout.current = setTimeout(() => setOpen(false), 150);
  }, []);

  const cancelClose = useCallback(() => {
    if (closeTimeout.current) {
      clearTimeout(closeTimeout.current);
      closeTimeout.current = null;
    }
  }, []);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <span
          className="text-fg-tertiary cursor-default text-xs underline decoration-dotted decoration-border underline-offset-4"
          onPointerEnter={() => {
            cancelClose();
            setOpen(true);
          }}
          onPointerLeave={scheduleClose}
        >
          {tags.length} tag{tags.length !== 1 ? "s" : ""}
        </span>
      </PopoverTrigger>
      <PopoverContent
        side="top"
        align="center"
        className="w-auto max-w-md p-3"
        onOpenAutoFocus={(e) => e.preventDefault()}
        onCloseAutoFocus={(e) => e.preventDefault()}
        onPointerEnter={cancelClose}
        onPointerLeave={scheduleClose}
      >
        <div className="space-y-1.5">
          {tags.map(([key, val]) => (
            <div key={key} className="flex gap-2 font-mono text-xs">
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-fg-secondary w-40 shrink-0 truncate">
                    {formatTagKey(key)}
                  </span>
                </TooltipTrigger>
                <TooltipContent side="left">
                  <span className="font-mono text-xs">{key}</span>
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-fg-primary truncate">{val}</span>
                </TooltipTrigger>
                <TooltipContent side="right">
                  <span className="font-mono text-xs break-all">{val}</span>
                </TooltipContent>
              </Tooltip>
            </div>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}
