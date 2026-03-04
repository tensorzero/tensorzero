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
  CommentFeedbackRow,
  DemonstrationFeedbackRow,
} from "~/types/tensorzero";
import { useConfig } from "~/context/config";
import { getFeedbackConfig } from "~/utils/config/feedback";
import type { FeedbackConfig } from "~/utils/config/feedback";
import { TableItemTime, TableItemShortUuid } from "~/components/ui/TableItems";
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
import { Sheet, SheetContent } from "~/components/ui/sheet";
import { CommentModal, DemonstrationModal } from "./FeedbackTableModal";
import { Skeleton } from "~/components/ui/skeleton";
import { Card, CardContent, CardHeader } from "~/components/ui/card";
import {
  parseInferenceOutput,
  isJsonOutput,
} from "~/utils/clickhouse/inference";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";

function formatTagKey(key: string): string {
  if (key.startsWith("tensorzero::")) {
    return key.slice("tensorzero::".length);
  }
  return key;
}

function MetricsTableHeaders() {
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

export function FeedbackTableHeaders() {
  return <MetricsTableHeaders />;
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
                <Skeleton className="h-4 w-24" />
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
  pagination?: React.ReactNode;
}

export default function FeedbackTable({
  feedback,
  feedbackBounds,
  latestByMetric,
  pagination,
}: FeedbackTableProps) {
  const config = useConfig();

  const { metrics, comments, demonstrations } = useMemo(() => {
    // Filter to latest-only when bounds are available
    let items = feedback;
    if (feedbackBounds && latestByMetric) {
      items = feedback.filter((item) => {
        if (item.type === "comment")
          return item.id === feedbackBounds.by_type.comment.last_id;
        if (item.type === "demonstration")
          return item.id === feedbackBounds.by_type.demonstration.last_id;
        return latestByMetric[item.metric_name] === item.id;
      });
    }

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

  if (feedback.length === 0) {
    return (
      <Table className="table-fixed">
        <MetricsTableHeaders />
        <TableBody>
          <TableEmptyState message="No feedback found" />
        </TableBody>
      </Table>
    );
  }

  return (
    <div className="space-y-6">
      {metrics.length > 0 && (
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
      )}

      {pagination}

      <FeedbackCard
        label="Comment"
        tags={comments[0]?.tags}
        timestamp={comments[0]?.timestamp}
        testId={
          comments[0] ? `feedback-row-${comments[0].id}` : "feedback-comment"
        }
        modal={
          comments[0] ? <CommentModal feedback={comments[0]} /> : undefined
        }
      >
        {comments[0]?.value ? (
          <p className="text-fg-primary line-clamp-2 text-sm">
            {comments[0].value}
          </p>
        ) : (
          <p className="text-fg-muted text-sm italic">No data</p>
        )}
      </FeedbackCard>

      <FeedbackCard
        label="Demonstration"
        tags={demonstrations[0]?.tags}
        timestamp={demonstrations[0]?.timestamp}
        testId={
          demonstrations[0]
            ? `feedback-row-${demonstrations[0].id}`
            : "feedback-demonstration"
        }
        modal={
          demonstrations[0] ? (
            <DemonstrationModal feedback={demonstrations[0]} />
          ) : undefined
        }
      >
        {demonstrations[0] ? (
          <DemonstrationPreview value={demonstrations[0].value} />
        ) : (
          <p className="text-fg-muted text-sm italic">No data</p>
        )}
      </FeedbackCard>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Metrics table row
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
  const displayMetric = getDisplayMetricName(item);

  const allTags = Object.entries(item.tags).filter(
    (entry): entry is [string, string] => typeof entry[1] === "string",
  );

  return (
    <TableRow data-testid={`feedback-row-${item.id}`}>
      <TableCell>
        <Tooltip>
          <TooltipTrigger asChild>
            <span
              className="cursor-default font-mono text-sm underline decoration-dotted decoration-border underline-offset-4"
              tabIndex={0}
            >
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
          <span className="text-fg-muted text-xs">&mdash;</span>
        )}
      </TableCell>
      <TableCell>
        <MetricConfigInfo feedbackConfig={feedbackConfig} />
      </TableCell>
      <TableCell>
        <span className="text-fg-tertiary flex items-center gap-1.5 text-xs">
          <TableItemTime timestamp={item.timestamp} />
          <span className="text-fg-muted">&middot;</span>
          <TableItemShortUuid id={item.id} />
        </span>
      </TableCell>
    </TableRow>
  );
}

// ---------------------------------------------------------------------------
// Metric config info (only for boolean/float metrics now)
// ---------------------------------------------------------------------------

interface MetricConfigInfoProps {
  feedbackConfig: FeedbackConfig | undefined;
}

function MetricConfigInfo({ feedbackConfig }: MetricConfigInfoProps) {
  if (
    !feedbackConfig ||
    feedbackConfig.type === "comment" ||
    feedbackConfig.type === "demonstration"
  ) {
    return <span className="text-fg-muted text-xs">&mdash;</span>;
  }

  const parts = [
    feedbackConfig.type,
    feedbackConfig.optimize,
    feedbackConfig.level,
  ];

  return (
    <span className="text-fg-tertiary text-xs">
      {parts.map((part, i) => (
        <span key={i}>
          {i > 0 && (
            <span className="text-fg-muted mx-1" aria-hidden>
              &middot;
            </span>
          )}
          {part}
        </span>
      ))}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Demonstration preview (renders parsed output in code block)
// ---------------------------------------------------------------------------

function DemonstrationPreview({ value }: { value: string }) {
  if (!value) {
    return <p className="text-fg-muted text-sm italic">No data</p>;
  }

  try {
    const parsedOutput = parseInferenceOutput(value);
    return (
      <div className="max-h-32 overflow-hidden">
        {isJsonOutput(parsedOutput) ? (
          <JsonOutputElement output={parsedOutput} />
        ) : (
          <ChatOutputElement output={parsedOutput} />
        )}
      </div>
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
  modal?: React.ReactNode;
}

function FeedbackCard({
  label,
  children,
  tags,
  timestamp,
  testId,
  modal,
}: FeedbackCardProps) {
  const [isSheetOpen, setIsSheetOpen] = useState(false);

  const allTags = tags
    ? Object.entries(tags).filter(
        (entry): entry is [string, string] => typeof entry[1] === "string",
      )
    : [];

  return (
    <>
      <Card
        data-testid={testId}
        className={`overflow-hidden${modal ? " hover:bg-bg-secondary cursor-pointer" : ""}`}
        onClick={modal ? () => setIsSheetOpen(true) : undefined}
      >
        <CardHeader className="bg-bg-secondary text-fg-tertiary flex h-10 justify-center border-b px-3 py-0 text-sm font-medium">
          {label}
        </CardHeader>
        <CardContent className="p-4">
          {children}
          {(allTags.length > 0 || timestamp) && (
            <div className="text-fg-tertiary mt-2 flex items-center gap-2 text-xs">
              {allTags.length > 0 && (
                <span onClick={(e) => e.stopPropagation()}>
                  <TagsPopover tags={allTags} />
                </span>
              )}
              {timestamp && (
                <span>
                  Last updated <TableItemTime timestamp={timestamp} />
                </span>
              )}
            </div>
          )}
        </CardContent>
      </Card>
      {modal && (
        <Sheet open={isSheetOpen} onOpenChange={setIsSheetOpen}>
          <SheetContent className="bg-bg-secondary overflow-y-auto p-0 sm:max-w-xl md:max-w-2xl lg:max-w-3xl">
            {modal}
          </SheetContent>
        </Sheet>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Tags popover (shared)
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
