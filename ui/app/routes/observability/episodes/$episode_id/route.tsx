import {
  countInferencesForEpisode,
  listInferencesWithPagination,
} from "~/utils/clickhouse/inference.server";
import {
  pollForFeedbackItem,
  queryLatestFeedbackIdByMetric,
} from "~/utils/clickhouse/feedback";
import { getNativeDatabaseClient } from "~/utils/tensorzero/native_client.server";
import type { Route } from "./+types/route";
import {
  data,
  isRouteErrorResponse,
  useNavigate,
  type RouteHandle,
  type ShouldRevalidateFunctionArgs,
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
import { useToast } from "~/hooks/use-toast";
import { useEffect, useState, useCallback } from "react";
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

/**
 * Prevent revalidation of this route when actions are submitted to API routes.
 * This is needed because:
 * 1. The InferencePreviewSheet submits feedback to /api/inference/:id
 * 2. The AddToDatasetButton submits to /api/datapoints
 * 3. By default, React Router revalidates all active loaders after any action
 * 4. We don't want to reload the entire episode page when these actions complete
 *    because the sheet handles its own data refresh
 */
export function shouldRevalidate({
  formAction,
  defaultShouldRevalidate,
}: ShouldRevalidateFunctionArgs) {
  if (
    formAction?.startsWith("/api/inference/") ||
    formAction?.startsWith("/api/datapoints")
  ) {
    return false;
  }
  return defaultShouldRevalidate;
}

export async function loader({ request, params }: Route.LoaderArgs) {
  const { episode_id } = params;
  const url = new URL(request.url);
  const beforeInference = url.searchParams.get("beforeInference");
  const afterInference = url.searchParams.get("afterInference");
  const beforeFeedback = url.searchParams.get("beforeFeedback");
  const afterFeedback = url.searchParams.get("afterFeedback");
  const limit = Number(url.searchParams.get("limit")) || 10;
  const newFeedbackId = url.searchParams.get("newFeedbackId");
  if (limit > 100) {
    throw data("Limit cannot exceed 100", { status: 400 });
  }

  const dbClient = await getNativeDatabaseClient();

  // If there is a freshly inserted feedback, ClickHouse may take some time to
  // update the feedback table and materialized views as it is eventually consistent.
  // In this case, we poll for the feedback item until it is found but time out and log a warning.
  // When polling for new feedback, we also need to query feedbackBounds and latestFeedbackByMetric
  // AFTER the polling completes to ensure the materialized views have caught up.
  const feedbackDataPromise = newFeedbackId
    ? pollForFeedbackItem(episode_id, newFeedbackId, limit)
    : dbClient.queryFeedbackByTargetId({
        target_id: episode_id,
        before: beforeFeedback || undefined,
        after: afterFeedback || undefined,
        limit,
      });

  let inferenceResult,
    feedbacks,
    feedbackBounds,
    num_inferences,
    num_feedbacks,
    latestFeedbackByMetric;

  if (newFeedbackId) {
    // When there's new feedback, wait for polling to complete before querying
    // feedbackBounds and latestFeedbackByMetric to ensure ClickHouse materialized views are updated
    [inferenceResult, feedbacks, num_inferences, num_feedbacks] =
      await Promise.all([
        listInferencesWithPagination({
          episode_id,
          before: beforeInference ?? undefined,
          after: afterInference ?? undefined,
          limit,
        }),
        feedbackDataPromise,
        countInferencesForEpisode(episode_id),
        dbClient.countFeedbackByTargetId({
          target_id: episode_id,
        }),
      ]);

    // Query these after polling completes to avoid race condition with materialized views
    [feedbackBounds, latestFeedbackByMetric] = await Promise.all([
      dbClient.queryFeedbackBoundsByTargetId({ target_id: episode_id }),
      queryLatestFeedbackIdByMetric({ target_id: episode_id }),
    ]);
  } else {
    // Normal case: execute all queries in parallel
    [
      inferenceResult,
      feedbacks,
      feedbackBounds,
      num_inferences,
      num_feedbacks,
      latestFeedbackByMetric,
    ] = await Promise.all([
      listInferencesWithPagination({
        episode_id,
        before: beforeInference ?? undefined,
        after: afterInference ?? undefined,
        limit,
      }),
      feedbackDataPromise,
      dbClient.queryFeedbackBoundsByTargetId({
        target_id: episode_id,
      }),
      countInferencesForEpisode(episode_id),
      dbClient.countFeedbackByTargetId({
        target_id: episode_id,
      }),
      queryLatestFeedbackIdByMetric({ target_id: episode_id }),
    ]);
  }
  if (inferenceResult.inferences.length === 0) {
    throw data(`No inferences found for episode ${episode_id}.`, {
      status: 404,
    });
  }

  return {
    episode_id,
    inferences: inferenceResult.inferences,
    hasNextInferencePage: inferenceResult.hasNextPage,
    hasPreviousInferencePage: inferenceResult.hasPreviousPage,
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
  const _action = formData.get("_action");

  switch (_action) {
    case "addFeedback": {
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

    case null:
      logger.error("No action provided");
      return data<ActionData>({ error: "No action provided" }, { status: 400 });

    default:
      logger.error(`Unknown action: ${_action}`);
      return data<ActionData>(
        { error: `Unknown action: ${_action}` },
        { status: 400 },
      );
  }
}

export default function InferencesPage({ loaderData }: Route.ComponentProps) {
  const {
    episode_id,
    inferences,
    hasNextInferencePage,
    hasPreviousInferencePage,
    feedbacks,
    feedbackBounds,
    num_inferences,
    num_feedbacks,
    newFeedbackId,
    latestFeedbackByMetric,
  } = loaderData;
  const navigate = useNavigate();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [openSheetInferenceId, setOpenSheetInferenceId] = useState<
    string | null
  >(null);

  const handleOpenSheet = useCallback((inferenceId: string) => {
    setOpenSheetInferenceId(inferenceId);
  }, []);

  const handleCloseSheet = useCallback(() => {
    setOpenSheetInferenceId(null);
  }, []);

  const topInference = inferences[0];
  const bottomInference = inferences[inferences.length - 1];
  const handleNextInferencePage = () => {
    if (!bottomInference) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("afterInference");
    searchParams.set("beforeInference", bottomInference.inference_id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousInferencePage = () => {
    if (!topInference) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("beforeInference");
    searchParams.set("afterInference", topInference.inference_id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

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
      const { dismiss } = toast.success({ title: "Feedback Added" });
      return () => dismiss({ immediate: true });
    }
    return;
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
          <EpisodeInferenceTable
            inferences={inferences}
            onOpenSheet={handleOpenSheet}
            onCloseSheet={handleCloseSheet}
            openSheetInferenceId={openSheetInferenceId}
          />
          <PageButtons
            onPreviousPage={handlePreviousInferencePage}
            onNextPage={handleNextInferencePage}
            disablePrevious={!hasPreviousInferencePage}
            disableNext={!hasNextInferencePage}
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
