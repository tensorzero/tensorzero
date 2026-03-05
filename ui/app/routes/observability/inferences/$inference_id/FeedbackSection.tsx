import { Suspense, useEffect } from "react";
import { Await, Link, useAsyncError, useNavigate } from "react-router";
import {
  TableErrorNotice,
  getErrorMessage,
} from "~/components/ui/error/ErrorContentPrimitives";
import { AlertCircle, MoveUpRight } from "lucide-react";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import PageButtons from "~/components/utils/PageButtons";
import FeedbackTable, {
  FeedbackTableSkeleton,
  filterToLatestFeedback,
} from "~/components/feedback/FeedbackTable";
import type { FeedbackData } from "./inference-data.server";

interface FeedbackSectionProps {
  promise: Promise<FeedbackData>;
  locationKey: string;
  count: number | undefined;
  onCountUpdate: (count: number) => void;
  episodeId: string;
  episodeFeedbackCount: Promise<number>;
  addFeedbackButton: React.ReactNode;
}

export function FeedbackSection({
  promise,
  locationKey,
  count,
  onCountUpdate,
  episodeId,
  episodeFeedbackCount,
  addFeedbackButton,
}: FeedbackSectionProps) {
  return (
    <SectionLayout>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <SectionHeader heading="Inference Feedback" count={count} />
        <div className="flex items-center gap-6">
          <Suspense fallback={null}>
            <Await resolve={episodeFeedbackCount} errorElement={null}>
              {(count) => (
                <EpisodeFeedbackNotice
                  episodeId={episodeId}
                  feedbackCount={count}
                />
              )}
            </Await>
          </Suspense>
          {addFeedbackButton}
        </div>
      </div>
      <Suspense key={`feedback-${locationKey}`} fallback={<FeedbackSkeleton />}>
        <Await resolve={promise} errorElement={<FeedbackError />}>
          {(data) => (
            <FeedbackContent data={data} onCountUpdate={onCountUpdate} />
          )}
        </Await>
      </Suspense>
    </SectionLayout>
  );
}

function FeedbackContent({
  data,
  onCountUpdate,
}: {
  data: FeedbackData;
  onCountUpdate: (count: number) => void;
}) {
  const { feedback, feedback_bounds, latestByMetric } = data;
  const navigate = useNavigate();

  const filteredCount = filterToLatestFeedback(
    feedback,
    feedback_bounds,
    latestByMetric,
  ).length;

  useEffect(() => {
    onCountUpdate(filteredCount);
  }, [filteredCount, onCountUpdate]);

  const topFeedback = feedback[0] as { id: string } | undefined;
  const bottomFeedback = feedback[feedback.length - 1] as
    | { id: string }
    | undefined;

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
    !feedback_bounds.last_id ||
    feedback_bounds.last_id === topFeedback.id;

  const disableNext =
    !bottomFeedback?.id ||
    !feedback_bounds.first_id ||
    feedback_bounds.first_id === bottomFeedback.id;

  const showPagination = !disablePrevious || !disableNext;

  return (
    <FeedbackTable
      feedback={feedback}
      feedbackBounds={feedback_bounds}
      latestByMetric={latestByMetric}
      pagination={
        showPagination ? (
          <PageButtons
            onPreviousPage={handlePreviousPage}
            onNextPage={handleNextPage}
            disablePrevious={disablePrevious}
            disableNext={disableNext}
          />
        ) : undefined
      }
    />
  );
}

function FeedbackSkeleton() {
  return <FeedbackTableSkeleton pagination={<PageButtons disabled />} />;
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

interface EpisodeFeedbackNoticeProps {
  episodeId: string;
  feedbackCount: number;
}

function EpisodeFeedbackNotice({
  episodeId,
  feedbackCount,
}: EpisodeFeedbackNoticeProps) {
  return (
    <Link
      to={`/observability/episodes/${episodeId}`}
      className="text-fg-muted hover:text-fg-secondary flex items-center gap-1.5 text-sm transition-colors"
    >
      Episode Feedback ({feedbackCount})
      <MoveUpRight className="h-3.5 w-3.5" />
    </Link>
  );
}
