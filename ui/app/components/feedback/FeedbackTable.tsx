import { useMemo, useState, useRef, useCallback } from "react";
import FeedbackValue from "~/components/feedback/FeedbackValue";
import { getMetricName } from "~/utils/clickhouse/helpers";
import type {
  FeedbackRow,
  FeedbackBounds,
  MetricConfig,
  CommentFeedbackRow,
  DemonstrationFeedbackRow,
} from "~/types/tensorzero";
import { useConfig } from "~/context/config";
import { getFeedbackConfig } from "~/utils/config/feedback";
import type { FeedbackConfig } from "~/utils/config/feedback";
import { TableItemTime } from "~/components/ui/TableItems";
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
import {
  parseInferenceOutput,
  isJsonOutput,
} from "~/utils/clickhouse/inference";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function formatTagKey(key: string): string {
  if (key.startsWith("tensorzero::")) {
    return key.slice("tensorzero::".length);
  }
  return key;
}

function filterStringTags(tags: Record<string, unknown>): [string, string][] {
  return Object.entries(tags).filter(
    (entry): entry is [string, string] => typeof entry[1] === "string",
  );
}

function NoData() {
  return <p className="text-fg-muted text-sm">No data</p>;
}

export function filterToLatestFeedback(
  feedback: FeedbackRow[],
  feedbackBounds?: FeedbackBounds,
  latestByMetric?: Record<string, string>,
): FeedbackRow[] {
  if (!feedbackBounds || !latestByMetric) return feedback;
  return feedback.filter((item) => {
    if (item.type === "comment")
      return item.id === feedbackBounds.by_type.comment.last_id;
    if (item.type === "demonstration")
      return item.id === feedbackBounds.by_type.demonstration.last_id;
    return latestByMetric[item.metric_name] === item.id;
  });
}

// ---------------------------------------------------------------------------
// Metrics table
// ---------------------------------------------------------------------------

function MetricsTableHeaders() {
  return (
    <TableHeader>
      <TableRow>
        <TableHead className="w-[40%]">Metric</TableHead>
        <TableHead className="w-[20%]">Value</TableHead>
        <TableHead className="w-[14%]">Tags</TableHead>
        <TableHead className="w-[26%]">Date</TableHead>
      </TableRow>
    </TableHeader>
  );
}

export function FeedbackTableSkeleton({
  pagination,
}: {
  pagination?: React.ReactNode;
}) {
  return (
    <div className="space-y-6">
      <Table className="table-fixed">
        <MetricsTableHeaders />
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
                <Skeleton className="h-4 w-28" />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {pagination}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main feedback component
// ---------------------------------------------------------------------------

interface FeedbackTableProps {
  feedback: FeedbackRow[];
  feedbackBounds?: FeedbackBounds;
  latestByMetric?: Record<string, string>;
  pagination?: React.ReactNode;
  showDemonstrations?: boolean;
}

export default function FeedbackTable({
  feedback,
  feedbackBounds,
  latestByMetric,
  pagination,
  showDemonstrations = true,
}: FeedbackTableProps) {
  const config = useConfig();

  const { metrics, comments, demonstrations } = useMemo(() => {
    const items = filterToLatestFeedback(
      feedback,
      feedbackBounds,
      latestByMetric,
    );

    return {
      metrics: items.filter((f) => f.type === "boolean" || f.type === "float"),
      comments: items.filter(
        (f): f is FeedbackRow & { type: "comment" } => f.type === "comment",
      ) as (CommentFeedbackRow & { type: "comment" })[],
      demonstrations: items.filter(
        (f): f is FeedbackRow & { type: "demonstration" } =>
          f.type === "demonstration",
      ) as (DemonstrationFeedbackRow & { type: "demonstration" })[],
    };
  }, [feedback, feedbackBounds, latestByMetric]);

  return (
    <div className="space-y-6">
      {metrics.length > 0 ? (
        <Table className="table-fixed">
          <MetricsTableHeaders />
          <TableBody>
            {metrics.map((item) => (
              <MetricRowItem
                key={item.id}
                item={item}
                metricConfig={config.metrics[getMetricName(item)]}
                feedbackConfig={getFeedbackConfig(getMetricName(item), config)}
              />
            ))}
          </TableBody>
        </Table>
      ) : (
        <Table className="table-fixed">
          <MetricsTableHeaders />
          <TableBody>
            <TableEmptyState message="No data" />
          </TableBody>
        </Table>
      )}

      {showDemonstrations && (
        <FeedbackCard
          label="Demonstration"
          tags={demonstrations[0]?.tags}
          timestamp={demonstrations[0]?.timestamp}
          testId={
            demonstrations[0]
              ? `feedback-row-${demonstrations[0].id}`
              : "feedback-demonstration"
          }
        >
          {demonstrations[0] ? (
            <div className="p-4">
              <DemonstrationPreview value={demonstrations[0].value} />
            </div>
          ) : (
            <div className="p-4">
              <NoData />
            </div>
          )}
        </FeedbackCard>
      )}

      <FeedbackCard
        label="Comment"
        tags={comments[0]?.tags}
        timestamp={comments[0]?.timestamp}
        testId={
          comments[0] ? `feedback-row-${comments[0].id}` : "feedback-comment"
        }
      >
        <div className="p-4">
          {comments[0]?.value ? (
            <p className="text-fg-primary text-sm">{comments[0].value}</p>
          ) : (
            <NoData />
          )}
        </div>
      </FeedbackCard>

      {pagination}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Metric table row
// ---------------------------------------------------------------------------

interface MetricRowItemProps {
  item: FeedbackRow;
  metricConfig: MetricConfig | undefined;
  feedbackConfig: FeedbackConfig | undefined;
}

function MetricRowItem({
  item,
  metricConfig,
  feedbackConfig,
}: MetricRowItemProps) {
  const metricName = getMetricName(item);
  const allTags = filterStringTags(item.tags);

  return (
    <TableRow data-testid={`feedback-row-${item.id}`}>
      <TableCell>
        <Tooltip>
          <TooltipTrigger asChild>
            <span
              className="cursor-default break-all font-mono text-sm underline decoration-dotted decoration-border underline-offset-4"
              tabIndex={0}
            >
              {metricName}
            </span>
          </TooltipTrigger>
          <TooltipContent sideOffset={5}>
            <MetricTooltipContent
              id={item.id}
              feedbackConfig={feedbackConfig}
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
          <span className="text-fg-muted text-xs">&mdash;</span>
        )}
      </TableCell>
      <TableCell>
        <span className="text-fg-tertiary text-xs">
          <TableItemTime timestamp={item.timestamp} />
        </span>
      </TableCell>
    </TableRow>
  );
}

interface MetricTooltipContentProps {
  id: string;
  feedbackConfig: FeedbackConfig | undefined;
}

function MetricTooltipContent({
  id,
  feedbackConfig,
}: MetricTooltipContentProps) {
  const configParts =
    feedbackConfig &&
    feedbackConfig.type !== "comment" &&
    feedbackConfig.type !== "demonstration"
      ? [feedbackConfig.type, feedbackConfig.optimize, feedbackConfig.level]
      : null;

  return (
    <div className="max-w-xs space-y-1.5 font-mono text-xs">
      <div className="break-all">ID: {id}</div>
      {configParts && <div>{configParts.join(" · ")}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Demonstration preview
// ---------------------------------------------------------------------------

function DemonstrationPreview({ value }: { value: string }) {
  if (!value) {
    return <NoData />;
  }

  try {
    const parsedOutput = parseInferenceOutput(value);
    return isJsonOutput(parsedOutput) ? (
      <JsonOutputElement output={parsedOutput} />
    ) : (
      <ChatOutputElement output={parsedOutput} />
    );
  } catch {
    return (
      <pre className="text-fg-primary line-clamp-3 whitespace-pre-wrap font-mono text-sm">
        {value}
      </pre>
    );
  }
}

// ---------------------------------------------------------------------------
// Feedback card (shared by comments and demonstrations)
// ---------------------------------------------------------------------------

interface FeedbackCardProps {
  label: string;
  children: React.ReactNode;
  tags?: Record<string, unknown>;
  timestamp?: string;
  testId: string;
}

function FeedbackCard({
  label,
  children,
  tags,
  timestamp,
  testId,
}: FeedbackCardProps) {
  const allTags = tags ? filterStringTags(tags) : [];

  return (
    <div data-testid={testId}>
      <div className="bg-bg-primary border-border overflow-hidden rounded-lg border [&_[data-testid=chat-output]]:!rounded-none [&_[data-testid=chat-output]]:!border-0 [&_[data-testid=chat-output]]:!p-0">
        <div className="bg-bg-secondary text-fg-tertiary flex h-10 items-center border-b px-3 text-sm font-medium">
          {label}
        </div>
        {children}
        {(allTags.length > 0 || timestamp) && (
          <div className="bg-bg-primary text-fg-tertiary flex items-center justify-between border-t px-3 py-2 text-xs">
            <span>
              {allTags.length > 0 ? <TagsPopover tags={allTags} /> : null}
            </span>
            {timestamp && (
              <span>
                Updated <TableItemTime timestamp={timestamp} />
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tag cell (shows tooltip only when display text differs from full text)
// ---------------------------------------------------------------------------

interface TagCellProps {
  displayText: string;
  fullText: string;
  className: string;
  tooltipSide: "left" | "right";
}

function TagCell({
  displayText,
  fullText,
  className,
  tooltipSide,
}: TagCellProps) {
  if (displayText === fullText) {
    return <span className={className}>{displayText}</span>;
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className={`${className} underline decoration-dotted decoration-border underline-offset-2`}
        >
          {displayText}
        </span>
      </TooltipTrigger>
      <TooltipContent side={tooltipSide}>
        <span className="break-all font-mono text-xs">{fullText}</span>
      </TooltipContent>
    </Tooltip>
  );
}

// ---------------------------------------------------------------------------
// Tags popover
// ---------------------------------------------------------------------------

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
              <TagCell
                displayText={formatTagKey(key)}
                fullText={key}
                className="text-fg-secondary w-40 shrink-0"
                tooltipSide="left"
              />
              <TagCell
                displayText={val}
                fullText={val}
                className="text-fg-primary"
                tooltipSide="right"
              />
            </div>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}
