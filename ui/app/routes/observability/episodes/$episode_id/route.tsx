import { listInferencesWithPagination } from "~/utils/clickhouse/inference.server";
import { pollForFeedbackItem } from "~/utils/clickhouse/feedback";
import { getNativeDatabaseClient } from "~/utils/tensorzero/native_client.server";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
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
import { useToast } from "~/hooks/use-toast";
import { Suspense, use, useEffect, useState, useCallback } from "react";
import { ActionBar } from "~/components/layout/ActionBar";
import { HumanFeedbackButton } from "~/components/feedback/HumanFeedbackButton";
import { HumanFeedbackModal } from "~/components/feedback/HumanFeedbackModal";
import { HumanFeedbackForm } from "~/components/feedback/HumanFeedbackForm";
import { useFetcherWithReset } from "~/hooks/use-fetcher-with-reset";
import { logger } from "~/utils/logger";
import type {
  StoredInference,
  FeedbackRow,
  FeedbackBounds,
} from "~/types/tensorzero";
import { Skeleton } from "~/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";

export type InferencesData = {
  inferences: StoredInference[];
  hasNextPage: boolean;
  hasPreviousPage: boolean;
};

export type FeedbackData = {
  feedbacks: FeedbackRow[];
  bounds: FeedbackBounds;
  latestFeedbackByMetric: Record<string, string>;
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

  const dbClient = await getNativeDatabaseClient();
  const tensorZeroClient = getTensorZeroClient();

  // Start count queries early - these will be streamed to section headers
  const numInferencesPromise = tensorZeroClient
    .getEpisodeInferenceCount(episode_id)
    .then((response) => response.inference_count);
  const numFeedbacksPromise = dbClient.countFeedbackByTargetId({
    target_id: episode_id,
  });

  // Stream inferences data - will be resolved in the component
  // Throws error if no inferences found (episode doesn't exist)
  const inferencesDataPromise: Promise<InferencesData> =
    listInferencesWithPagination({
      episode_id,
      before: beforeInference ?? undefined,
      after: afterInference ?? undefined,
      limit,
    }).then((result) => {
      if (result.inferences.length === 0) {
        throw Error(`Episode not found`);
      }
      return {
        inferences: result.inferences,
        hasNextPage: result.hasNextPage,
        hasPreviousPage: result.hasPreviousPage,
      };
    });

  // Stream feedback data - will be resolved in the component
  // If there is a freshly inserted feedback, ClickHouse may take some time to
  // update the feedback table and materialized views as it is eventually consistent.
  // In this case, we poll for the feedback item until it is found but time out and log a warning.
  // When polling for new feedback, we also need to query feedbackBounds and latestFeedbackByMetric
  // AFTER the polling completes to ensure the materialized views have caught up.
  const feedbackDataPromise: Promise<FeedbackData> = newFeedbackId
    ? // Sequential case: poll first, then query bounds/metrics
      pollForFeedbackItem(episode_id, newFeedbackId, limit).then(
        async (feedbacks) => {
          const [bounds, latestFeedbackByMetric] = await Promise.all([
            dbClient.queryFeedbackBoundsByTargetId({ target_id: episode_id }),
            tensorZeroClient.getLatestFeedbackIdByMetric(episode_id),
          ]);
          return { feedbacks, bounds, latestFeedbackByMetric };
        },
      )
    : // Normal case: execute all queries in parallel
      Promise.all([
        tensorZeroClient.getFeedbackByTargetId(episode_id, {
          before: beforeFeedback || undefined,
          after: afterFeedback || undefined,
          limit,
        }),
        dbClient.queryFeedbackBoundsByTargetId({ target_id: episode_id }),
        tensorZeroClient.getLatestFeedbackIdByMetric(episode_id),
      ]).then(([feedbacks, bounds, latestFeedbackByMetric]) => ({
        feedbacks,
        bounds,
        latestFeedbackByMetric,
      }));

  return {
    episode_id,
    inferencesData: inferencesDataPromise,
    feedbackData: feedbackDataPromise,
    // Stream counts to section headers
    num_inferences: numInferencesPromise,
    num_feedbacks: numFeedbacksPromise,
    newFeedbackId,
  };
}

/** Response type from /api/feedback endpoint */
type FeedbackActionData =
  | { redirectTo: string; error?: never }
  | { error: string; redirectTo?: never };

function InferencePagination({ data }: { data: Promise<InferencesData> }) {
  const { inferences, hasNextPage, hasPreviousPage } = use(data);
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

function FeedbackTableSkeleton() {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>ID</TableHead>
          <TableHead>Metric</TableHead>
          <TableHead>Value</TableHead>
          <TableHead>Tags</TableHead>
          <TableHead>Time</TableHead>
        </TableRow>
      </TableHeader>
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
  );
}

function FeedbackSection({ data }: { data: Promise<FeedbackData> }) {
  const { feedbacks, bounds, latestFeedbackByMetric } = use(data);
  const navigate = useNavigate();

  const topFeedback = feedbacks[0] as { id: string } | undefined;
  const bottomFeedback = feedbacks[feedbacks.length - 1] as
    | { id: string }
    | undefined;

  const handleNextPage = () => {
    if (!bottomFeedback?.id) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("afterFeedback");
    searchParams.set("beforeFeedback", bottomFeedback.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  const handlePreviousPage = () => {
    if (!topFeedback?.id) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("beforeFeedback");
    searchParams.set("afterFeedback", topFeedback.id);
    navigate(`?${searchParams.toString()}`, { preventScrollReset: true });
  };

  // These are swapped because the table is sorted in descending order
  const disablePrevious =
    !topFeedback?.id || !bounds.last_id || bounds.last_id === topFeedback.id;

  const disableNext =
    !bottomFeedback?.id ||
    !bounds.first_id ||
    bounds.first_id === bottomFeedback.id;

  return (
    <>
      <FeedbackTable
        feedback={feedbacks}
        latestCommentId={bounds.by_type.comment.last_id!}
        latestDemonstrationId={bounds.by_type.demonstration.last_id!}
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

export default function EpisodeDetailPage({
  loaderData,
}: Route.ComponentProps) {
  const {
    episode_id,
    inferencesData,
    feedbackData,
    num_inferences,
    num_feedbacks,
    newFeedbackId,
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
          <Suspense
            fallback={
              <PageButtons
                onPreviousPage={() => {}}
                onNextPage={() => {}}
                disablePrevious
                disableNext
              />
            }
          >
            <InferencePagination data={inferencesData} />
          </Suspense>
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
          <Suspense
            fallback={
              <>
                <FeedbackTableSkeleton />
                <PageButtons
                  onPreviousPage={() => {}}
                  onNextPage={() => {}}
                  disablePrevious
                  disableNext
                />
              </>
            }
          >
            <FeedbackSection data={feedbackData} />
          </Suspense>
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
