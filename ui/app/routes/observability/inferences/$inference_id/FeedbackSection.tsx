import { Suspense, useEffect } from "react";
import { Await, Link, useAsyncError, useNavigate } from "react-router";
import {
  TableErrorNotice,
  getErrorMessage,
} from "~/components/ui/error/ErrorContentPrimitives";
import { AlertCircle, ArrowRight } from "lucide-react";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import PageButtons from "~/components/utils/PageButtons";
import FeedbackTable, {
  FeedbackTableSkeleton,
} from "~/components/feedback/FeedbackTable";
import type { FeedbackData } from "./inference-data.server";

interface FeedbackSectionProps {
  promise: Promise<FeedbackData>;
  locationKey: string;
  count: number | undefined;
  onCountUpdate: (count: number) => void;
  episodeId: string;
  episodeFeedbackCount: Promise<number>;
}

export function FeedbackSection({
  promise,
  locationKey,
  count,
  onCountUpdate,
  episodeId,
  episodeFeedbackCount,
}: FeedbackSectionProps) {
  return (
    <SectionLayout>
      <SectionHeader
        heading="Feedback"
        count={count}
        badge={{
          name: "inference",
          tooltip:
            "This table only includes inference-level feedback. To see episode-level feedback, open the detail page for that episode.",
        }}
      />
      <Suspense key={`feedback-${locationKey}`} fallback={<FeedbackSkeleton />}>
        <Await resolve={promise} errorElement={<FeedbackError />}>
          {(data) => (
            <FeedbackContent data={data} onCountUpdate={onCountUpdate} />
          )}
        </Await>
      </Suspense>
      <Suspense fallback={null}>
        <Await resolve={episodeFeedbackCount} errorElement={null}>
          {(count) =>
            count > 0 ? (
              <EpisodeFeedbackNotice
                episodeId={episodeId}
                feedbackCount={count}
              />
            ) : null
          }
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

  useEffect(() => {
    onCountUpdate(feedback.length);
  }, [feedback.length, onCountUpdate]);

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

  return (
    <>
      <FeedbackTable
        feedback={feedback}
        feedbackBounds={feedback_bounds}
        latestByMetric={latestByMetric}
        pagination={
          <PageButtons
            onPreviousPage={handlePreviousPage}
            onNextPage={handleNextPage}
            disablePrevious={disablePrevious}
            disableNext={disableNext}
          />
        }
      />
    </>
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
      className="text-fg-muted hover:text-fg-secondary mt-2 flex items-center gap-1.5 text-sm transition-colors"
    >
      This episode has {feedbackCount} feedback{" "}
      {feedbackCount === 1 ? "entry" : "entries"}
      <ArrowRight className="h-3.5 w-3.5" />
    </Link>
  );
}
