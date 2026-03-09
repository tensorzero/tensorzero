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

// Parses evaluator metric names like
// "tensorzero::evaluation_name::xxx::evaluator_name::yyy"
// into a short display name ("yyy") with evaluation context.
const EVAL_PREFIX = "tensorzero::evaluation_name::";
const EVALUATOR_SEPARATOR = "::evaluator_name::";

interface ParsedMetricName {
  displayName: string;
  fullName: string;
  isEvaluator: boolean;
  evaluationName?: string;
}

function parseMetricName(metricName: string): ParsedMetricName {
  if (metricName.startsWith(EVAL_PREFIX)) {
    const rest = metricName.slice(EVAL_PREFIX.length);
    const sepIndex = rest.indexOf(EVALUATOR_SEPARATOR);
    if (sepIndex !== -1) {
      return {
        displayName: rest.slice(sepIndex + EVALUATOR_SEPARATOR.length),
        fullName: metricName,
        isEvaluator: true,
        evaluationName: rest.slice(0, sepIndex),
      };
    }
  }
  return { displayName: metricName, fullName: metricName, isEvaluator: false };
}

export function MetricFeedbackTableHeaders() {
  return (
    <TableHeader>
      <TableRow>
        <TableHead className="w-[52%]">Metric</TableHead>
        <TableHead className="w-[15%]">Value</TableHead>
        <TableHead className="w-[11%]">Tags</TableHead>
        <TableHead className="w-[22%]">Date</TableHead>
      </TableRow>
    </TableHeader>
  );
}

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
  const parsed = parseMetricName(metricName);
  const allTags = filterStringTags(item.tags);

  return (
    <TableRow data-testid={`feedback-row-${item.id}`}>
      <TableCell>
        <Tooltip>
          <TooltipTrigger asChild>
            <span
              className="flex cursor-default items-center gap-1.5 font-mono text-sm"
              tabIndex={0}
            >
              <span className="truncate underline decoration-dotted decoration-border underline-offset-4">
                {parsed.displayName}
              </span>
              {parsed.isEvaluator && (
                <span className="bg-bg-tertiary text-fg-tertiary shrink-0 rounded px-1 py-0.5 font-sans text-[10px] font-medium leading-none">
                  eval
                </span>
              )}
            </span>
          </TooltipTrigger>
          <TooltipContent sideOffset={5}>
            <MetricTooltipContent
              id={item.id}
              fullName={parsed.fullName}
              evaluationName={parsed.evaluationName}
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
  fullName: string;
  evaluationName?: string;
  feedbackConfig: FeedbackConfig | undefined;
}

function MetricTooltipContent({
  id,
  fullName,
  evaluationName,
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
      <div className="break-all">{fullName}</div>
      {evaluationName && (
        <div className="text-fg-tertiary">evaluation: {evaluationName}</div>
      )}
      <div className="break-all">ID: {id}</div>
      {configParts && <div>{configParts.join(" · ")}</div>}
    </div>
  );
}
