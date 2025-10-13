import {
  countInferencesForEpisode,
  queryInferenceTableBoundsByEpisodeId,
  queryInferenceTableByEpisodeId,
} from "~/utils/clickhouse/inference.server";
import {
  pollForFeedbackItem,
  queryLatestFeedbackIdByMetric,
} from "~/utils/clickhouse/feedback";
import { getDatabaseClient } from "~/utils/clickhouse/client.server";
import type { Route } from "./+types/route";
import {
  data,
  isRouteErrorResponse,
  useNavigate,
  type RouteHandle,
} from "react-router";
import EpisodeInferenceTable from "./EpisodeInferenceTable";
import FeedbackTable from "~/components/feedback/FeedbackTable";
import PageButtons from "~/components/utils/PageButtons";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
  SectionsGroup,
  SectionHeader,
} from "~/components/layout/PageLayout";
import { addHumanFeedback } from "~/utils/tensorzero.server";
import { Toaster } from "~/components/ui/toaster";
import { useToast } from "~/hooks/use-toast";
import { useEffect, useState } from "react";
import { ActionBar } from "~/components/layout/ActionBar";
import { HumanFeedbackButton } from "~/components/feedback/HumanFeedbackButton";
import { HumanFeedbackModal } from "~/components/feedback/HumanFeedbackModal";
import { HumanFeedbackForm } from "~/components/feedback/HumanFeedbackForm";
import { useFetcherWithReset } from "~/hooks/use-fetcher-with-reset";
import { logger } from "~/utils/logger";
import { isTensorZeroServerError } from "~/utils/tensorzero";

export const handle: RouteHandle = {
  crumb: (match) => [{ label: match.params.episode_id!, isIdentifier: true }],
};

export async function loader({ request, params }: Route.LoaderArgs) {
  const { episode_id } = params;
  const url = new URL(request.url);
  const beforeInference = url.searchParams.get("beforeInference");
  const afterInference = url.searchParams.get("afterInference");
  const beforeFeedback = url.searchParams.get("beforeFeedback");
  const afterFeedback = url.searchParams.get("afterFeedback");
  const pageSize = Number(url.searchParams.get("pageSize")) || 10;
  const newFeedbackId = url.searchParams.get("newFeedbackId");
  if (pageSize > 100) {
    throw data("Page size cannot exceed 100", { status: 400 });
  }

  const dbClient = await getDatabaseClient();

  // If there is a freshly inserted feedback, ClickHouse may take some time to
  // update the feedback table as it is eventually consistent.
  // In this case, we poll for the feedback item until it is found but time out and log a warning.
  const feedbackDataPromise = newFeedbackId
    ? pollForFeedbackItem(episode_id, newFeedbackId, pageSize)
    : dbClient.queryFeedbackByTargetId(
        episode_id,
        beforeFeedback || undefined,
        afterFeedback || undefined,
        pageSize,
      );

  const [
    inferences,
    inference_bounds,
    feedbacks,
    feedbackBounds,
    num_inferences,
    num_feedbacks,
    latestFeedbackByMetric,
  ] = await Promise.all([
    queryInferenceTableByEpisodeId({
      episode_id,
      before: beforeInference ?? undefined,
      after: afterInference ?? undefined,
      page_size: pageSize,
    }),
    queryInferenceTableBoundsByEpisodeId({
      episode_id,
    }),
    feedbackDataPromise,
    dbClient.queryFeedbackBoundsByTargetId(episode_id),
    countInferencesForEpisode(episode_id),
    dbClient.countFeedbackByTargetId(episode_id),
    queryLatestFeedbackIdByMetric({ target_id: episode_id }),
  ]);
  if (inferences.length === 0) {
    throw data(`No inferences found for episode ${episode_id}.`, {
      status: 404,
    });
  }

  return {
    episode_id,
    inferences,
    inference_bounds,
    feedbacks,
    feedbackBounds,
    num_inferences,
    num_feedbacks,
    newFeedbackId,
    latestFeedbackByMetric,
  };
}

type ActionData =
  | { redirectTo: string; error?: never }
  | { error: string; redirectTo?: never };

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();

  try {
    const response = await addHumanFeedback(formData);
    const url = new URL(request.url);
    url.searchParams.delete("beforeFeedback");
    url.searchParams.delete("afterFeedback");
    url.searchParams.set("newFeedbackId", response.feedback_id);
    return data<ActionData>({ redirectTo: url.pathname + url.search });
  } catch (error) {
    if (isTensorZeroServerError(error)) {
      return data<ActionData>(
        { error: error.message },
        { status: error.status },
      );
    }
    return data<ActionData>(
      { error: "Unknown server error. Try again." },
      { status: 500 },
    );
  }
}

export default function InferencesPage({ loaderData }: Route.ComponentProps) {
  const {
    episode_id,
    inferences,
    inference_bounds,
    feedbacks,
    feedbackBounds,
    num_inferences,
    num_feedbacks,
    newFeedbackId,
    latestFeedbackByMetric,
  } = loaderData;
  const navigate = useNavigate();
  const [isModalOpen, setIsModalOpen] = useState(false);

  const topInference = inferences[0];
  const bottomInference = inferences[inferences.length - 1];
  const handleNextInferencePage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("afterInference");
    searchParams.set("beforeInference", bottomInference.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousInferencePage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("beforeInference");
    searchParams.set("afterInference", topInference.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };
  // These are swapped because the table is sorted in descending order
  const disablePreviousInferencePage =
    inference_bounds?.last_id === topInference.id;
  const disableNextInferencePage =
    inference_bounds?.first_id === bottomInference.id;

  const topFeedback = feedbacks[0] as { id: string } | undefined;
  const bottomFeedback = feedbacks[feedbacks.length - 1] as
    | { id: string }
    | undefined;

  const handleNextFeedbackPage = () => {
    if (!bottomFeedback?.id) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("afterFeedback");
    searchParams.set("beforeFeedback", bottomFeedback.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousFeedbackPage = () => {
    if (!topFeedback?.id) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("beforeFeedback");
    searchParams.set("afterFeedback", topFeedback.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const { toast } = useToast();
  useEffect(() => {
    if (newFeedbackId) {
      toast({
        title: "Feedback Added",
      });
    }
  }, [newFeedbackId, toast]);
  // These are swapped because the table is sorted in descending order
  const disablePreviousFeedbackPage =
    !topFeedback?.id ||
    !feedbackBounds.last_id ||
    feedbackBounds.last_id === topFeedback.id;

  const disableNextFeedbackPage =
    !bottomFeedback?.id ||
    !feedbackBounds.first_id ||
    feedbackBounds.first_id === bottomFeedback.id;

  const humanFeedbackFetcher = useFetcherWithReset<typeof action>();
  const formError =
    humanFeedbackFetcher.state === "idle"
      ? (humanFeedbackFetcher.data?.error ?? null)
      : null;
  useEffect(() => {
    const currentState = humanFeedbackFetcher.state;
    const data = humanFeedbackFetcher.data;
    if (currentState === "idle" && data?.redirectTo) {
      navigate(data.redirectTo);
      setIsModalOpen(false);
    }
  }, [humanFeedbackFetcher.data, humanFeedbackFetcher.state, navigate]);

  return (
    <PageLayout>
      <PageHeader label="Episode" name={episode_id}>
        <ActionBar>
          <HumanFeedbackModal
            isOpen={isModalOpen}
            onOpenChange={(isOpen) => {
              if (humanFeedbackFetcher.state !== "idle") {
                return;
              }

              if (!isOpen) {
                humanFeedbackFetcher.reset();
              }
              setIsModalOpen(isOpen);
            }}
            trigger={<HumanFeedbackButton />}
          >
            <humanFeedbackFetcher.Form method="post">
              <HumanFeedbackForm
                episodeId={episode_id}
                formError={formError}
                isSubmitting={
                  humanFeedbackFetcher.state === "submitting" ||
                  humanFeedbackFetcher.state === "loading"
                }
              />
            </humanFeedbackFetcher.Form>
          </HumanFeedbackModal>
        </ActionBar>
      </PageHeader>

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Inferences" count={num_inferences} />
          <EpisodeInferenceTable inferences={inferences} />
          <PageButtons
            onPreviousPage={handlePreviousInferencePage}
            onNextPage={handleNextInferencePage}
            disablePrevious={disablePreviousInferencePage}
            disableNext={disableNextInferencePage}
          />
        </SectionLayout>

        <SectionLayout>
          <SectionHeader
            heading="Feedback"
            count={num_feedbacks}
            badge={{
              name: "episode",
              tooltip:
                "This table only includes episode-level feedback. To see inference-level feedback, open the detail page for that inference.",
            }}
          />
          <FeedbackTable
            feedback={feedbacks}
            latestCommentId={feedbackBounds.by_type.comment.last_id!}
            latestDemonstrationId={
              feedbackBounds.by_type.demonstration.last_id!
            }
            latestFeedbackIdByMetric={latestFeedbackByMetric}
          />
          <PageButtons
            onPreviousPage={handlePreviousFeedbackPage}
            onNextPage={handleNextFeedbackPage}
            disablePrevious={disablePreviousFeedbackPage}
            disableNext={disableNextFeedbackPage}
          />
        </SectionLayout>
      </SectionsGroup>
      <Toaster />
    </PageLayout>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  logger.error(error);

  if (isRouteErrorResponse(error)) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">
          {error.status} {error.statusText}
        </h1>
        <p>{error.data}</p>
      </div>
    );
  } else if (error instanceof Error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 text-red-500">
        <h1 className="text-2xl font-bold">Error</h1>
        <p>{error.message}</p>
      </div>
    );
  } else {
    return (
      <div className="flex h-screen items-center justify-center text-red-500">
        <h1 className="text-2xl font-bold">Unknown Error</h1>
      </div>
    );
  }
}
