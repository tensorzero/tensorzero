import {
  queryInferenceById,
  queryModelInferencesByInferenceId,
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
  Link,
  useNavigate,
  type RouteHandle,
} from "react-router";
import PageButtons from "~/components/utils/PageButtons";
import { addHumanFeedback } from "~/utils/tensorzero.server";
import { handleAddToDatasetAction } from "~/utils/dataset.server";
import { useEffect } from "react";
import type { ReactNode } from "react";
import { PageHeader, PageLayout } from "~/components/layout/PageLayout";
import { useToast } from "~/hooks/use-toast";
import { logger } from "~/utils/logger";
import { isTensorZeroServerError } from "~/utils/tensorzero";
import { getUsedVariants } from "~/utils/clickhouse/function";
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

  const dbClient = await getNativeDatabaseClient();

  const inferencePromise = queryInferenceById(inference_id);
  const modelInferencesPromise =
    queryModelInferencesByInferenceId(inference_id);
  const demonstrationFeedbackPromise =
    dbClient.queryDemonstrationFeedbackByInferenceId({
      inference_id,
      limit: 1, // Only need to know if *any* exist
    });
  // If there is a freshly inserted feedback, ClickHouse may take some time to
  // update the feedback table and materialized views as it is eventually consistent.
  // In this case, we poll for the feedback item until it is found but eventually time out and log a warning.
  // When polling for new feedback, we also need to query feedbackBounds and latestFeedbackByMetric
  // AFTER the polling completes to ensure the materialized views have caught up.
  const feedbackDataPromise = newFeedbackId
    ? pollForFeedbackItem(inference_id, newFeedbackId, limit)
    : dbClient.queryFeedbackByTargetId({
        target_id: inference_id,
        before: beforeFeedback || undefined,
        after: afterFeedback || undefined,
        limit,
      });

  // --- Execute promises concurrently (with special handling for new feedback) ---

  let inference,
    model_inferences,
    demonstration_feedback,
    feedback_bounds,
    feedback,
    latestFeedbackByMetric;

  if (newFeedbackId) {
    // When there's new feedback, wait for polling to complete before querying
    // feedbackBounds and latestFeedbackByMetric to ensure ClickHouse materialized views are updated
    [inference, model_inferences, demonstration_feedback, feedback] =
      await Promise.all([
        inferencePromise,
        modelInferencesPromise,
        demonstrationFeedbackPromise,
        feedbackDataPromise,
      ]);

    // Query these after polling completes to avoid race condition with materialized views
    [feedback_bounds, latestFeedbackByMetric] = await Promise.all([
      dbClient.queryFeedbackBoundsByTargetId({ target_id: inference_id }),
      queryLatestFeedbackIdByMetric({ target_id: inference_id }),
    ]);
  } else {
    // Normal case: execute all queries in parallel
    [
      inference,
      model_inferences,
      demonstration_feedback,
      feedback_bounds,
      feedback,
      latestFeedbackByMetric,
    ] = await Promise.all([
      inferencePromise,
      modelInferencesPromise,
      demonstrationFeedbackPromise,
      dbClient.queryFeedbackBoundsByTargetId({ target_id: inference_id }),
      feedbackDataPromise,
      queryLatestFeedbackIdByMetric({ target_id: inference_id }),
    ]);
  }

  // --- Process results ---

  if (!inference) {
    throw data(`No inference found for id ${inference_id}.`, {
      status: 404,
    });
  }

  const usedVariants =
    inference.function_name === DEFAULT_FUNCTION
      ? await getUsedVariants(inference.function_name)
      : [];

  return {
    inference,
    model_inferences,
    usedVariants,
    feedback,
    feedback_bounds,
    hasDemonstration: demonstration_feedback.length > 0,
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
    case "addToDataset": {
      return handleAddToDatasetAction(formData);
    }
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
        { error: "Unknown server action" },
        { status: 400 },
      );
  }
}

export default function InferencePage({ loaderData }: Route.ComponentProps) {
  const {
    inference,
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
    model_inferences,
    feedback,
    feedback_bounds,
    hasDemonstration,
    latestFeedbackByMetric,
    usedVariants,
  };

  return (
    <PageLayout>
      <InferenceDetailContent
        data={inferenceData}
        onFeedbackAdded={(redirectUrl) => {
          if (redirectUrl) {
            navigate(redirectUrl);
          }
        }}
        feedbackFooter={
          <PageButtons
            onNextPage={handleNextFeedbackPage}
            onPreviousPage={handlePreviousFeedbackPage}
            disableNext={disableNextFeedbackPage}
            disablePrevious={disablePreviousFeedbackPage}
          />
        }
        renderHeader={({ basicInfo, actionBar }) => (
          <PageHeader label="Inference" name={inference.id}>
            {basicInfo}
            {actionBar}
          </PageHeader>
        )}
      />
    </PageLayout>
  );
}

function getUserFacingError(error: unknown): {
  heading: string;
  message: ReactNode;
} {
  if (isRouteErrorResponse(error)) {
    switch (error.status) {
      case 400:
        return {
          heading: `${error.status}: Bad Request`,
          message: "Please try again later.",
        };
      case 401:
        return {
          heading: `${error.status}: Unauthorized`,
          message: "You do not have permission to access this resource.",
        };
      case 403:
        return {
          heading: `${error.status}: Forbidden`,
          message: "You do not have permission to access this resource.",
        };
      case 404:
        return {
          heading: `${error.status}: Not Found`,
          message:
            "The requested resource was not found. Please check the URL and try again.",
        };
      case 500:
      default:
        return {
          heading: "An unknown error occurred",
          message: "Please try again later.",
        };
    }
  }
  return {
    heading: "An unknown error occurred",
    message: "Please try again later.",
  };
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  useEffect(() => {
    logger.error(error);
  }, [error]);
  const { heading, message } = getUserFacingError(error);
  return (
    <div className="flex flex-col items-center justify-center md:h-full">
      <div className="mt-8 flex flex-col items-center justify-center gap-2 rounded-xl bg-red-50 p-6 md:mt-0">
        <h1 className="text-2xl font-bold">{heading}</h1>
        {typeof message === "string" ? <p>{message}</p> : message}
        <Link
          to={`/observability/inferences`}
          className="font-bold text-red-800 hover:text-red-600"
        >
          Go back &rarr;
        </Link>
      </div>
    </div>
  );
}
