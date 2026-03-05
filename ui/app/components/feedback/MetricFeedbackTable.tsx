import FeedbackValue from "~/components/feedback/FeedbackValue";
import { getMetricName } from "~/utils/clickhouse/helpers";
import type { FeedbackRow, MetricConfig } from "~/types/tensorzero";
import { useConfig } from "~/context/config";
import {
  getFeedbackConfig,
  type FeedbackConfig,
} from "~/utils/config/feedback";
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
  TagsPopover,
  filterStringTags,
} from "~/components/feedback/TagsPopover";

// ---------------------------------------------------------------------------
// Headers (shared with skeleton)
// ---------------------------------------------------------------------------

export function MetricFeedbackTableHeaders() {
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

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

interface MetricFeedbackTableProps {
  metrics: FeedbackRow[];
}

export function MetricFeedbackTable({ metrics }: MetricFeedbackTableProps) {
  const config = useConfig();

  if (metrics.length === 0) {
    return (
      <Table className="table-fixed">
        <MetricFeedbackTableHeaders />
        <TableBody>
          <TableEmptyState message="No data" />
        </TableBody>
      </Table>
    );
  }

  return (
    <Table className="table-fixed">
      <MetricFeedbackTableHeaders />
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
  );
}

// ---------------------------------------------------------------------------
// Row
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

// ---------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------

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
