import { useMemo } from "react";
import type { FeedbackRow, FeedbackBounds } from "~/types/tensorzero";
import { Table, TableBody, TableCell, TableRow } from "~/components/ui/table";
import { Skeleton } from "~/components/ui/skeleton";
import {
  MetricFeedbackTable,
  MetricFeedbackTableHeaders,
} from "~/components/feedback/MetricFeedbackTable";
import {
  CommentCard,
  DemonstrationCard,
} from "~/components/feedback/FeedbackCard";

export type FeedbackData = {
  feedback: FeedbackRow[];
  feedbackBounds: FeedbackBounds;
  latestFeedbackByMetric: Record<string, string>;
};

export function filterToLatestFeedback(
  feedback: FeedbackRow[],
  feedbackBounds?: FeedbackBounds,
  latestFeedbackByMetric?: Record<string, string>,
): FeedbackRow[] {
  if (!feedbackBounds || !latestFeedbackByMetric) return feedback;
  return feedback.filter((item) => {
    if (item.type === "comment") {
      const lastId = feedbackBounds.by_type.comment.last_id;
      return lastId === undefined || item.id === lastId;
    }
    if (item.type === "demonstration") {
      const lastId = feedbackBounds.by_type.demonstration.last_id;
      return lastId === undefined || item.id === lastId;
    }
    return latestFeedbackByMetric[item.metric_name] === item.id;
  });
}

function CardSkeleton({ label }: { label: string }) {
  return (
    <div className="bg-bg-primary border-border overflow-hidden rounded-lg border">
      <div className="bg-bg-secondary text-fg-tertiary flex h-10 items-center border-b px-3 text-sm font-medium">
        {label}
      </div>
      <div className="p-4">
        <Skeleton className="h-4 w-48" />
      </div>
    </div>
  );
}

export function FeedbackDisplaySkeleton({
  pagination,
  showDemonstrations = true,
}: {
  pagination?: React.ReactNode;
  showDemonstrations?: boolean;
}) {
  return (
    <div className="space-y-6">
      <Table className="table-fixed">
        <MetricFeedbackTableHeaders />
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
      {showDemonstrations && <CardSkeleton label="Demonstration" />}
      <CardSkeleton label="Comment" />
      {pagination}
    </div>
  );
}

interface FeedbackDisplayProps {
  feedback: FeedbackRow[];
  pagination?: React.ReactNode;
  showDemonstrations?: boolean;
}

// Pure renderer — expects already-filtered feedback.
// Filtering is done by FeedbackSection (for full pages) or the API route
// (for the preview sheet in InferenceDetailContent).
export default function FeedbackDisplay({
  feedback,
  pagination,
  showDemonstrations = true,
}: FeedbackDisplayProps) {
  const { metrics, comments, demonstrations } = useMemo(() => {
    return {
      metrics: feedback.filter(
        (f) => f.type === "boolean" || f.type === "float",
      ),
      comments: feedback.filter(
        (f): f is FeedbackRow & { type: "comment" } => f.type === "comment",
      ),
      demonstrations: feedback.filter(
        (f): f is FeedbackRow & { type: "demonstration" } =>
          f.type === "demonstration",
      ),
    };
  }, [feedback]);

  return (
    <div className="space-y-6">
      <MetricFeedbackTable metrics={metrics} />
      {showDemonstrations && (
        <DemonstrationCard demonstration={demonstrations[0]} />
      )}
      <CommentCard comment={comments[0]} />
      {pagination}
    </div>
  );
}
