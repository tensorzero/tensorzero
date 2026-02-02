import { Suspense, useEffect } from "react";
import { Await, useAsyncError, useNavigate } from "react-router";
import { Skeleton } from "~/components/ui/skeleton";
import { Table, TableBody, TableCell, TableRow } from "~/components/ui/table";
import {
  TableErrorNotice,
  getErrorMessage,
} from "~/components/ui/error/ErrorContentPrimitives";
import { AlertCircle } from "lucide-react";
import { SectionHeader, SectionLayout } from "~/components/layout/PageLayout";
import PageButtons from "~/components/utils/PageButtons";
import FeedbackTable, {
  FeedbackTableHeaders,
} from "~/components/feedback/FeedbackTable";
import type { FeedbackData } from "./inference-data.server";

interface FeedbackSectionProps {
  promise: Promise<FeedbackData>;
  locationKey: string;
  count: number | undefined;
  onCountUpdate: (count: number) => void;
}

export function FeedbackSection({
  promise,
  locationKey,
  count,
  onCountUpdate,
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
  const { feedback, feedback_bounds, latestFeedbackByMetric } = data;
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
        latestCommentId={feedback_bounds.by_type.comment.last_id!}
        latestDemonstrationId={feedback_bounds.by_type.demonstration.last_id!}
        latestFeedbackIdByMetric={latestFeedbackByMetric}
      />
      <PageButtons
        onPreviousPage={handlePreviousPage}
        onNextPage={handleNextPage}
        disablePrevious={disablePrevious}
        disableNext={disableNext}
      />
    </>
  );
}

function FeedbackSkeleton() {
  return (
    <>
      <Table>
        <FeedbackTableHeaders />
        <TableBody>
          {Array.from({ length: 5 }).map((_, i) => (
            <TableRow key={i}>
              <TableCell>
                <Skeleton className="h-4 w-24" />
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
      <PageButtons disabled />
    </>
  );
}

function FeedbackError() {
  const error = useAsyncError();
  const message = getErrorMessage({
    error,
    defaultMessage: "Failed to load feedback",
  });

  return (
    <>
      <Table>
        <FeedbackTableHeaders />
        <TableBody>
          <TableRow>
            <TableCell colSpan={5}>
              <TableErrorNotice
                icon={AlertCircle}
                title="Error loading data"
                description={message}
              />
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
      <PageButtons disabled />
    </>
  );
}
