import { Suspense, useEffect, useState, useCallback } from "react";
import {
  Await,
  data,
  useLocation,
  useNavigate,
  type RouteHandle,
  type ShouldRevalidateFunctionArgs,
} from "react-router";
import { listInferencesWithPagination } from "~/utils/clickhouse/inference.server";
import { pollForFeedbackItem } from "~/utils/clickhouse/feedback";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { Route } from "./+types/route";
import type { FeedbackData } from "~/components/feedback/FeedbackDisplay";
import { FeedbackSection } from "~/components/feedback/FeedbackSection";
import EpisodeInferenceTable from "./EpisodeInferenceTable";
import PageButtons from "~/components/utils/PageButtons";
import { AskAutopilotButton } from "~/components/autopilot/AskAutopilotButton";
import {
  PageHeader,
  PageLayout,
  SectionLayout,
  SectionsGroup,
  SectionHeader,
  Breadcrumbs,
} from "~/components/layout/PageLayout";
import { useToast } from "~/hooks/use-toast";
import { ActionBar } from "~/components/layout/ActionBar";
import { HumanFeedbackButton } from "~/components/feedback/HumanFeedbackButton";
import { HumanFeedbackModal } from "~/components/feedback/HumanFeedbackModal";
import { HumanFeedbackForm } from "~/components/feedback/HumanFeedbackForm";
import { useFetcherWithReset } from "~/hooks/use-fetcher-with-reset";
import { HelpTooltip } from "~/components/ui/HelpTooltip";
import type { StoredInference } from "~/types/tensorzero";
import type { FeedbackActionData } from "~/routes/api/feedback/route";

export type InferencesData = {
  inferences: StoredInference[];
  hasNextPage: boolean;
  hasPreviousPage: boolean;
};

export const handle: RouteHandle = {
  crumb: (match) => [{ label: match.params.episode_id!, isIdentifier: true }],
};

/**
 * Prevent revalidation of this route when actions are submitted to API routes.
 * This is needed because:
 * 1. The InferencePreviewSheet submits feedback to /api/feedback
 * 2. The AddToDatasetButton submits to /api/datasets/datapoints/from-inference
 * 3. By default, React Router revalidates all active loaders after any action
 * 4. We don't want to reload the entire episode page when these actions complete
 *    because the sheet handles its own data refresh
 */
export function shouldRevalidate({
  formAction,
  defaultShouldRevalidate,
}: ShouldRevalidateFunctionArgs) {
  if (
    formAction?.startsWith("/api/feedback") ||
    formAction?.startsWith("/api/datasets/datapoints/from-inference")
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

  const tensorZeroClient = getTensorZeroClient();

  // Check if episode exists by getting inference count upfront
  const inferenceCountResponse =
    await tensorZeroClient.getEpisodeInferenceCount(episode_id);
  const numInferences = inferenceCountResponse.inference_count;

  // Convert to number for comparison - JSON returns number, types may vary
  if (Number(numInferences) === 0) {
    throw data(`Episode "${episode_id}" not found`, { status: 404 });
  }

  // Count is already fetched, wrap in resolved promise for streaming
  const numInferencesPromise = Promise.resolve(numInferences);

  // Stream inferences data - will be resolved in the component
  const inferencesDataPromise: Promise<InferencesData> =
    listInferencesWithPagination({
      episode_id,
      before: beforeInference ?? undefined,
      after: afterInference ?? undefined,
      limit,
    }).then((result) => ({
      inferences: result.inferences,
      hasNextPage: result.hasNextPage,
      hasPreviousPage: result.hasPreviousPage,
    }));

  // Stream feedback data - will be resolved in the component
  // If there is a freshly inserted feedback, ClickHouse may take some time to
  // update the feedback table and materialized views as it is eventually consistent.
  // In this case, we poll for the feedback item until it is found but time out and log a warning.
  // When polling for new feedback, we also need to query feedbackBounds
  // AFTER the polling completes to ensure the materialized views have caught up.
  const feedbackDataPromise: Promise<FeedbackData> = newFeedbackId
    ? // Sequential case: poll first, then query bounds/metrics
      pollForFeedbackItem(episode_id, newFeedbackId, limit).then(
        async (feedback) => {
          const [feedbackBounds, latestFeedbackByMetric] = await Promise.all([
            tensorZeroClient.getFeedbackBoundsByTargetId(episode_id),
            tensorZeroClient.getLatestFeedbackIdByMetric(episode_id),
          ]);
          return {
            feedback,
            feedbackBounds,
            latestFeedbackByMetric,
          };
        },
      )
    : // Normal case: execute all queries in parallel
      Promise.all([
        tensorZeroClient.getFeedbackByTargetId(episode_id, {
          before: beforeFeedback || undefined,
          after: afterFeedback || undefined,
          limit,
        }),
        tensorZeroClient.getFeedbackBoundsByTargetId(episode_id),
        tensorZeroClient.getLatestFeedbackIdByMetric(episode_id),
      ]).then(([feedback, feedbackBounds, latestFeedbackByMetric]) => ({
        feedback,
        feedbackBounds,
        latestFeedbackByMetric,
      }));

  return {
    episode_id,
    inferencesData: inferencesDataPromise,
    feedbackData: feedbackDataPromise,
    // Stream counts to section headers
    num_inferences: numInferencesPromise,
    newFeedbackId,
  };
}

function InferencePaginationContent({ data }: { data: InferencesData }) {
  const { inferences, hasNextPage, hasPreviousPage } = data;
  const navigate = useNavigate();

  const topInference = inferences.at(0);
  const bottomInference = inferences.at(-1);

  const handleNextPage = () => {
    if (!bottomInference) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("afterInference");
    searchParams.set("beforeInference", bottomInference.inference_id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousPage = () => {
    if (!topInference) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("beforeInference");
    searchParams.set("afterInference", topInference.inference_id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  return (
    <PageButtons
      onPreviousPage={handlePreviousPage}
      onNextPage={handleNextPage}
      disablePrevious={!hasPreviousPage}
      disableNext={!hasNextPage}
    />
  );
}

export default function EpisodeDetailPage({
  loaderData,
}: Route.ComponentProps) {
  const {
    episode_id,
    inferencesData,
    feedbackData,
    num_inferences,
    newFeedbackId,
  } = loaderData;
  const location = useLocation();
  const navigate = useNavigate();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [feedbackCount, setFeedbackCount] = useState<number | undefined>(
    undefined,
  );
  const [openSheetInferenceId, setOpenSheetInferenceId] = useState<
    string | null
  >(null);

  const handleOpenSheet = useCallback((inferenceId: string) => {
    setOpenSheetInferenceId(inferenceId);
  }, []);

  const handleCloseSheet = useCallback(() => {
    setOpenSheetInferenceId(null);
  }, []);

  // Reset feedback count on navigation so stale count doesn't linger
  useEffect(() => {
    setFeedbackCount(undefined);
  }, [location.key]);

  const { toast } = useToast();
  useEffect(() => {
    if (newFeedbackId) {
      const { dismiss } = toast.success({ title: "Feedback Added" });
      return () => dismiss({ immediate: true });
    }
    return;
  }, [newFeedbackId, toast]);

  const humanFeedbackFetcher = useFetcherWithReset<FeedbackActionData>();
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
      <PageHeader
        eyebrow={
          <Breadcrumbs
            segments={[{ label: "Episodes", href: "/observability/episodes" }]}
          />
        }
        name={episode_id}
      >
        <ActionBar>
          <AskAutopilotButton message={`Episode ID: ${episode_id}\n\n`} />
        </ActionBar>
      </PageHeader>

      <SectionsGroup>
        <SectionLayout>
          <SectionHeader heading="Inferences" count={num_inferences} />
          <EpisodeInferenceTable
            data={inferencesData}
            onOpenSheet={handleOpenSheet}
            onCloseSheet={handleCloseSheet}
            openSheetInferenceId={openSheetInferenceId}
          />
          <Suspense fallback={<PageButtons disabled />}>
            <Await
              resolve={inferencesData}
              errorElement={<PageButtons disabled />}
            >
              {(resolvedData) => (
                <InferencePaginationContent data={resolvedData} />
              )}
            </Await>
          </Suspense>
        </SectionLayout>

        <SectionLayout>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <SectionHeader
              heading="Episode Feedback"
              count={feedbackCount}
              help={
                <HelpTooltip>
                  This table only includes episode-level feedback. To see
                  inference-level feedback, open the detail page for that
                  inference.
                </HelpTooltip>
              }
            />
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
              <humanFeedbackFetcher.Form method="post" action="/api/feedback">
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
          </div>
          <FeedbackSection
            promise={feedbackData}
            locationKey={location.key}
            onCountUpdate={setFeedbackCount}
            showDemonstrations={false}
          />
        </SectionLayout>
      </SectionsGroup>
    </PageLayout>
  );
}
