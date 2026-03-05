import { Suspense, useEffect } from "react";
import { Await, useAsyncError, useNavigate } from "react-router";
import { AlertCircle } from "lucide-react";
import {
  TableErrorNotice,
  getErrorMessage,
} from "~/components/ui/error/ErrorContentPrimitives";
import PageButtons from "~/components/utils/PageButtons";
import FeedbackDisplay, {
  FeedbackDisplaySkeleton,
  filterToLatestFeedback,
  type FeedbackData,
} from "~/components/feedback/FeedbackDisplay";

interface FeedbackSectionProps {
  promise: Promise<FeedbackData>;
  locationKey: string;
  onCountUpdate: (count: number) => void;
  showDemonstrations?: boolean;
}

export function FeedbackSection({
  promise,
  locationKey,
  onCountUpdate,
  showDemonstrations = true,
}: FeedbackSectionProps) {
  return (
    <Suspense
      key={`feedback-${locationKey}`}
      fallback={
        <FeedbackDisplaySkeleton
          pagination={<PageButtons disabled />}
          showDemonstrations={showDemonstrations}
        />
      }
    >
      <Await resolve={promise} errorElement={<FeedbackError />}>
        {(data) => (
          <FeedbackContent
            data={data}
            onCountUpdate={onCountUpdate}
            showDemonstrations={showDemonstrations}
          />
        )}
      </Await>
    </Suspense>
  );
}

function FeedbackContent({
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

  return (
    <FeedbackDisplay
      feedback={filteredFeedback}
      showDemonstrations={showDemonstrations}
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
