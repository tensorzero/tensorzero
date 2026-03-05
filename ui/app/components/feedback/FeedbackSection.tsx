import { Suspense, useEffect } from "react";
import { Await, useAsyncError, useNavigate } from "react-router";
import { AlertCircle } from "lucide-react";
import type { FeedbackRow } from "~/types/tensorzero";
import { filterToLatestFeedback, type FeedbackData } from "~/utils/feedback";
import { Table, TableBody, TableCell, TableRow } from "~/components/ui/table";
import { Skeleton } from "~/components/ui/skeleton";
import {
  TableErrorNotice,
  getErrorMessage,
} from "~/components/ui/error/ErrorContentPrimitives";
import PageButtons from "~/components/utils/PageButtons";
import {
  MetricFeedbackTable,
  MetricFeedbackTableHeaders,
} from "~/components/feedback/MetricFeedbackTable";
import {
  CommentCard,
  DemonstrationCard,
} from "~/components/feedback/FeedbackCard";

// --- Skeleton ---

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

export function FeedbackSectionSkeleton({
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

// --- Streaming internals ---

function StreamingContent({
  data,
  onCountUpdate,
  showDemonstrations,
}: {
  data: FeedbackData;
  onCountUpdate: (count: number) => void;
  showDemonstrations: boolean;
}) {
  const { feedback, feedbackBounds, latestFeedbackByMetric } = data;
  const navigate = useNavigate();

  const filteredFeedback = filterToLatestFeedback(
    feedback,
    feedbackBounds,
    latestFeedbackByMetric,
  );

  useEffect(() => {
    onCountUpdate(filteredFeedback.length);
  }, [filteredFeedback.length, onCountUpdate]);

  // Pagination uses the raw (unfiltered) feedback array since cursor pagination
  // is based on position in the full result set, not the filtered view.
  const topFeedback = feedback[0];
  const bottomFeedback = feedback[feedback.length - 1];

  const handleNextPage = () => {
    if (!bottomFeedback?.id) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("afterFeedback");
    searchParams.delete("newFeedbackId");
    searchParams.set("beforeFeedback", bottomFeedback.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousPage = () => {
    if (!topFeedback?.id) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("beforeFeedback");
    searchParams.delete("newFeedbackId");
    searchParams.set("afterFeedback", topFeedback.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const disablePrevious =
    !topFeedback?.id ||
    !feedbackBounds.last_id ||
    feedbackBounds.last_id === topFeedback.id;

  const disableNext =
    !bottomFeedback?.id ||
    !feedbackBounds.first_id ||
    feedbackBounds.first_id === bottomFeedback.id;

  const showPagination = !disablePrevious || !disableNext;

  const metrics = filteredFeedback.filter(
    (f) => f.type === "boolean" || f.type === "float",
  );
  const comments = filteredFeedback.filter(
    (f): f is FeedbackRow & { type: "comment" } => f.type === "comment",
  );
  const demonstrations = filteredFeedback.filter(
    (f): f is FeedbackRow & { type: "demonstration" } =>
      f.type === "demonstration",
  );

  return (
    <div className="space-y-6">
      <MetricFeedbackTable metrics={metrics} />
      {showDemonstrations && (
        <DemonstrationCard demonstration={demonstrations[0]} />
      )}
      <CommentCard comment={comments[0]} />
      {showPagination && (
        <PageButtons
          onPreviousPage={handlePreviousPage}
          onNextPage={handleNextPage}
          disablePrevious={disablePrevious}
          disableNext={disableNext}
        />
      )}
    </div>
  );
}

function FeedbackError() {
  const error = useAsyncError();
  const message = getErrorMessage({
    error,
    fallback: "Failed to load feedback",
  });

  return (
    <div className="rounded-lg border py-8">
      <TableErrorNotice
        icon={AlertCircle}
        title="Error loading data"
        description={message}
      />
    </div>
  );
}

// --- Main component ---

interface StreamingFeedbackSectionProps {
  promise: Promise<FeedbackData>;
  locationKey: string;
  onCountUpdate: (count: number) => void;
  showDemonstrations?: boolean;
}

interface DirectFeedbackSectionProps {
  feedback: FeedbackRow[];
  showDemonstrations?: boolean;
  pagination?: React.ReactNode;
}

type FeedbackSectionProps =
  | StreamingFeedbackSectionProps
  | DirectFeedbackSectionProps;

export function FeedbackSection(props: FeedbackSectionProps) {
  if ("promise" in props) {
    const {
      promise,
      locationKey,
      onCountUpdate,
      showDemonstrations = true,
    } = props;
    return (
      <Suspense
        key={`feedback-${locationKey}`}
        fallback={
          <FeedbackSectionSkeleton
            pagination={<PageButtons disabled />}
            showDemonstrations={showDemonstrations}
          />
        }
      >
        <Await resolve={promise} errorElement={<FeedbackError />}>
          {(data) => (
            <StreamingContent
              data={data}
              onCountUpdate={onCountUpdate}
              showDemonstrations={showDemonstrations}
            />
          )}
        </Await>
      </Suspense>
    );
  }

  const { feedback, showDemonstrations = true, pagination } = props;
  const metrics = feedback.filter(
    (f) => f.type === "boolean" || f.type === "float",
  );
  const comments = feedback.filter(
    (f): f is FeedbackRow & { type: "comment" } => f.type === "comment",
  );
  const demonstrations = feedback.filter(
    (f): f is FeedbackRow & { type: "demonstration" } =>
      f.type === "demonstration",
  );

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
