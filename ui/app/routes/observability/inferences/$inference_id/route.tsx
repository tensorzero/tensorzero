import { pollForFeedbackItem } from "~/utils/clickhouse/feedback";
import { getTensorZeroClient } from "~/utils/tensorzero.server";
import {
  resolveModelInferences,
  loadFileDataForStoredInput,
} from "~/utils/resolve.server";
import type { Route } from "./+types/route";
import { data, useNavigate, type RouteHandle } from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import { useEffect } from "react";
import {
  PageHeader,
  PageLayout,
  Breadcrumbs,
} from "~/components/layout/PageLayout";
import { useToast } from "~/hooks/use-toast";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import {
  InferenceDetailContent,
  type InferenceDetailData,
} from "~/components/inference/InferenceDetailContent";

export const handle: RouteHandle = {
  crumb: (match) => [{ label: match.params.inference_id!, isIdentifier: true }],
};

export async function loader({ request, params }: Route.LoaderArgs) {
  const { inference_id } = params;
  const url = new URL(request.url);
  const newFeedbackId = url.searchParams.get("newFeedbackId");
  const beforeFeedback = url.searchParams.get("beforeFeedback");
  const afterFeedback = url.searchParams.get("afterFeedback");
  const limit = Number(url.searchParams.get("limit")) || 10;

  if (limit > 100) {
    throw data("Limit cannot exceed 100", { status: 400 });
  }

  // --- Define all promises, conditionally choosing the feedback promise ---

  const tensorZeroClient = getTensorZeroClient();

  const inferencesPromise = tensorZeroClient.getInferences({
    ids: [inference_id],
    output_source: "inference",
  });
  const modelInferencesPromise = tensorZeroClient
    .getModelInferences(inference_id)
    .then((response) => resolveModelInferences(response.model_inferences));
  const demonstrationFeedbackPromise =
    tensorZeroClient.getDemonstrationFeedback(
      inference_id,
      { limit: 1 }, // Only need to know if *any* exist
    );
  // If there is a freshly inserted feedback, ClickHouse may take some time to
  // update the feedback table and materialized views as it is eventually consistent.
  // In this case, we poll for the feedback item until it is found but eventually time out and log a warning.
  // When polling for new feedback, we also need to query feedbackBounds and latestFeedbackByMetric
  // AFTER the polling completes to ensure the materialized views have caught up.
  const feedbackDataPromise = newFeedbackId
    ? pollForFeedbackItem(inference_id, newFeedbackId, limit)
    : tensorZeroClient.getFeedbackByTargetId(inference_id, {
        before: beforeFeedback || undefined,
        after: afterFeedback || undefined,
        limit,
      });

  // --- Execute promises concurrently (with special handling for new feedback) ---

  let inferences,
    model_inferences,
    demonstration_feedback,
    feedback_bounds,
    feedback,
    latestFeedbackByMetric;

  if (newFeedbackId) {
    // When there's new feedback, wait for polling to complete before querying
    // feedbackBounds and latestFeedbackByMetric to ensure ClickHouse materialized views are updated
    [inferences, model_inferences, demonstration_feedback, feedback] =
      await Promise.all([
        inferencesPromise,
        modelInferencesPromise,
        demonstrationFeedbackPromise,
        feedbackDataPromise,
      ]);

    // Query these after polling completes to avoid race condition with materialized views
    [feedback_bounds, latestFeedbackByMetric] = await Promise.all([
      tensorZeroClient.getFeedbackBoundsByTargetId(inference_id),
      tensorZeroClient.getLatestFeedbackIdByMetric(inference_id),
    ]);
  } else {
    // Normal case: execute all queries in parallel
    [
      inferences,
      model_inferences,
      demonstration_feedback,
      feedback_bounds,
      feedback,
      latestFeedbackByMetric,
    ] = await Promise.all([
      inferencesPromise,
      modelInferencesPromise,
      demonstrationFeedbackPromise,
      tensorZeroClient.getFeedbackBoundsByTargetId(inference_id),
      feedbackDataPromise,
      tensorZeroClient.getLatestFeedbackIdByMetric(inference_id),
    ]);
  }

  // --- Process results ---

  if (inferences.inferences.length !== 1) {
    throw data(`No inference found for id ${inference_id}.`, {
      status: 404,
    });
  }
  const inference = inferences.inferences[0];

  const usedVariants =
    inference.function_name === DEFAULT_FUNCTION
      ? await tensorZeroClient.getUsedVariants(inference.function_name)
      : [];
  const resolvedInput = await loadFileDataForStoredInput(inference.input);

  return {
    inference,
    resolvedInput,
    model_inferences,
    usedVariants,
    feedback,
    feedback_bounds,
    hasDemonstration: demonstration_feedback.length > 0,
    newFeedbackId,
    latestFeedbackByMetric,
  };
}

export default function InferencePage({ loaderData }: Route.ComponentProps) {
  const {
    inference,
    resolvedInput,
    model_inferences,
    usedVariants,
    feedback,
    feedback_bounds,
    hasDemonstration,
    newFeedbackId,
    latestFeedbackByMetric,
  } = loaderData;
  const navigate = useNavigate();
  const { toast } = useToast();

  // Show toast when feedback is successfully added
  useEffect(() => {
    if (newFeedbackId) {
      const { dismiss } = toast.success({ title: "Feedback Added" });
      return () => dismiss({ immediate: true });
    }
    return;
  }, [newFeedbackId, toast]);

  // Feedback pagination
  const topFeedback = feedback[0] as { id: string } | undefined;
  const bottomFeedback = feedback[feedback.length - 1] as
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

  // These are swapped because the table is sorted in descending order
  const disablePreviousFeedbackPage =
    !topFeedback?.id ||
    !feedback_bounds.last_id ||
    feedback_bounds.last_id === topFeedback.id;

  const disableNextFeedbackPage =
    !bottomFeedback?.id ||
    !feedback_bounds.first_id ||
    feedback_bounds.first_id === bottomFeedback.id;

  // Build the data object for InferenceDetailContent
  const inferenceData: InferenceDetailData = {
    inference,
    input: resolvedInput,
    model_inferences,
    feedback,
    feedback_bounds,
    hasDemonstration,
    latestFeedbackByMetric,
    usedVariants,
  };

  // Handle feedback added callback - extract newFeedbackId from the API redirect URL
  // and navigate to the page URL with the newFeedbackId
  const handleFeedbackAdded = (redirectUrl?: string) => {
    if (redirectUrl) {
      // redirectUrl is like /api/inference/{id}?newFeedbackId={feedbackId}
      // Extract the newFeedbackId and navigate to the current page with it
      const url = new URL(redirectUrl, window.location.origin);
      const newFeedbackIdParam = url.searchParams.get("newFeedbackId");
      if (newFeedbackIdParam) {
        const currentUrl = new URL(window.location.href);
        currentUrl.searchParams.delete("beforeFeedback");
        currentUrl.searchParams.delete("afterFeedback");
        currentUrl.searchParams.set("newFeedbackId", newFeedbackIdParam);
        navigate(currentUrl.pathname + currentUrl.search);
      }
    }
  };

  return (
    <PageLayout>
      <InferenceDetailContent
        data={inferenceData}
        onFeedbackAdded={handleFeedbackAdded}
        feedbackFooter={
          <PageButtons
            onNextPage={handleNextFeedbackPage}
            onPreviousPage={handlePreviousFeedbackPage}
            disableNext={disableNextFeedbackPage}
            disablePrevious={disablePreviousFeedbackPage}
          />
        }
        renderHeader={({ basicInfo, actionBar }) => (
          <PageHeader
            eyebrow={
              <Breadcrumbs
                segments={[
                  { label: "Inferences", href: "/observability/inferences" },
                ]}
              />
            }
            name={inference.inference_id}
          >
            {basicInfo}
            {actionBar}
          </PageHeader>
        )}
      />
    </PageLayout>
  );
}
