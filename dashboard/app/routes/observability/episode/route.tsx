import {
  countInferencesForEpisode,
  queryInferenceTableBoundsByEpisodeId,
  queryInferenceTableByEpisodeId,
} from "~/utils/clickhouse/inference";
import {
  countFeedbackByTargetId,
  queryFeedbackBoundsByTargetId,
  queryFeedbackByTargetId,
} from "~/utils/clickhouse/feedback";
import type { Route } from "./+types/route";
import { data, isRouteErrorResponse, useNavigate } from "react-router";
import { Badge } from "~/components/ui/badge";
import EpisodeInferenceTable from "./EpisodeInferenceTable";
import {
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { Tooltip } from "~/components/ui/tooltip";
import EpisodeFeedbackTable from "./EpisodeFeedbackTable";
import PageButtons from "~/components/utils/PageButtons";

export async function loader({ request, params }: Route.LoaderArgs) {
  const { episode_id } = params;
  const url = new URL(request.url);
  const beforeInference = url.searchParams.get("beforeInference");
  const afterInference = url.searchParams.get("afterInference");
  const beforeFeedback = url.searchParams.get("beforeFeedback");
  const afterFeedback = url.searchParams.get("afterFeedback");
  const pageSize = Number(url.searchParams.get("pageSize")) || 10;
  if (pageSize > 100) {
    throw data("Page size cannot exceed 100", { status: 400 });
  }

  const [
    inferences,
    inference_bounds,
    feedbacks,
    feedback_bounds,
    num_inferences,
    num_feedbacks,
  ] = await Promise.all([
    queryInferenceTableByEpisodeId({
      episode_id,
      before: beforeInference || undefined,
      after: afterInference || undefined,
      page_size: pageSize,
    }),
    queryInferenceTableBoundsByEpisodeId({
      episode_id,
    }),
    queryFeedbackByTargetId({
      target_id: episode_id,
      before: beforeFeedback || undefined,
      after: afterFeedback || undefined,
      page_size: pageSize,
    }),
    queryFeedbackBoundsByTargetId({
      target_id: episode_id,
    }),
    countInferencesForEpisode(episode_id),
    countFeedbackByTargetId(episode_id),
  ]);

  return {
    episode_id,
    inferences,
    inference_bounds,
    feedbacks,
    feedback_bounds,
    num_inferences,
    num_feedbacks,
  };
}

export default function InferencesPage({ loaderData }: Route.ComponentProps) {
  const {
    episode_id,
    inferences,
    inference_bounds,
    feedbacks,
    feedback_bounds,
    num_inferences,
    num_feedbacks,
  } = loaderData;
  const navigate = useNavigate();

  const topInference = inferences[0];
  const bottomInference = inferences[inferences.length - 1];
  const handleNextInferencePage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("afterInference");
    searchParams.set("beforeInference", bottomInference.id);
    navigate(`?${searchParams.toString()}`);
  };

  const handlePreviousInferencePage = () => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("beforeInference");
    searchParams.set("afterInference", topInference.id);
    navigate(`?${searchParams.toString()}`);
  };
  // These are swapped because the table is sorted in descending order
  const disablePreviousInferencePage =
    inference_bounds.last_id === topInference.id;
  const disableNextInferencePage =
    inference_bounds.first_id === bottomInference.id;

  const topFeedback = feedbacks[0] as { id: string } | undefined;
  const bottomFeedback = feedbacks[feedbacks.length - 1] as
    | { id: string }
    | undefined;

  const handleNextFeedbackPage = () => {
    if (!bottomFeedback?.id) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("afterFeedback");
    searchParams.set("beforeFeedback", bottomFeedback.id);
    navigate(`?${searchParams.toString()}`);
  };

  const handlePreviousFeedbackPage = () => {
    if (!topFeedback?.id) return;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.delete("beforeFeedback");
    searchParams.set("afterFeedback", topFeedback.id);
    navigate(`?${searchParams.toString()}`);
  };

  // These are swapped because the table is sorted in descending order
  const disablePreviousFeedbackPage =
    !topFeedback?.id ||
    !feedback_bounds.last_id ||
    feedback_bounds.last_id === topFeedback.id;

  const disableNextFeedbackPage =
    !bottomFeedback?.id ||
    !feedback_bounds.first_id ||
    feedback_bounds.first_id === bottomFeedback.id;

  return (
    <div className="container mx-auto px-4 py-8">
      <h2 className="mb-4 text-2xl font-semibold">
        Episode{" "}
        <code className="rounded bg-gray-100 p-1 text-2xl">{episode_id}</code>
      </h2>
      <div className="mb-6 h-px w-full bg-gray-200"></div>

      <div>
        <h3 className="mb-2 flex items-center gap-2 text-xl font-semibold">
          Inferences
          <Badge variant="secondary">Count: {num_inferences}</Badge>
        </h3>

        <EpisodeInferenceTable inferences={inferences} />
        <PageButtons
          onPreviousPage={handlePreviousInferencePage}
          onNextPage={handleNextInferencePage}
          disablePrevious={disablePreviousInferencePage}
          disableNext={disableNextInferencePage}
        />
      </div>

      <div className="mt-8">
        <h3 className="mb-2 flex items-center gap-2 text-xl font-semibold">
          Feedback
          <TooltipProvider>
            <Tooltip delayDuration={0}>
              <TooltipTrigger asChild>
                <Badge variant="outline" className="px-2 py-0.5 text-xs">
                  episode
                </Badge>
              </TooltipTrigger>
              <TooltipContent>
                <p className="max-w-xs">
                  This table only includes episode-level feedback. To see
                  inference-level feedback, open the detail page for that
                  inference.
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <Badge variant="secondary">Count: {num_feedbacks}</Badge>
        </h3>
        <EpisodeFeedbackTable feedback={feedbacks} />
        <PageButtons
          onPreviousPage={handlePreviousFeedbackPage}
          onNextPage={handleNextFeedbackPage}
          disablePrevious={disablePreviousFeedbackPage}
          disableNext={disableNextFeedbackPage}
        />
      </div>
    </div>
  );
}
export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  console.error(error);

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
