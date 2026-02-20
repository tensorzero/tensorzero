import { data, type LoaderFunctionArgs } from "react-router";
import { listInferencesWithPagination } from "~/utils/clickhouse/inference.server";
import { pollForFeedbackItem } from "~/utils/clickhouse/feedback";
import { logger } from "~/utils/logger";
import type { EpisodeDetailData } from "~/components/episode/EpisodeDetailContent";
import { getTensorZeroClient } from "~/utils/tensorzero.server";

export async function loader({
  request,
  params,
}: LoaderFunctionArgs): Promise<Response> {
  const { episode_id } = params;
  const url = new URL(request.url);
  const newFeedbackId = url.searchParams.get("newFeedbackId");

  if (!episode_id) {
    throw data("Episode ID is required", { status: 400 });
  }

  try {
    const client = getTensorZeroClient();

    const inferenceCountResponse =
      await client.getEpisodeInferenceCount(episode_id);
    const numInferences = inferenceCountResponse.inference_count;

    if (Number(numInferences) === 0) {
      throw data(`Episode "${episode_id}" not found`, { status: 404 });
    }

    const inferencesPromise = listInferencesWithPagination({
      episode_id,
      limit: 10,
    });

    const numFeedbacksPromise = client.countFeedbackByTargetId(episode_id);

    const feedbackDataPromise = newFeedbackId
      ? pollForFeedbackItem(episode_id, newFeedbackId, 10).then(
          async (feedbacks) => {
            const [bounds, latestFeedbackByMetric] = await Promise.all([
              client.getFeedbackBoundsByTargetId(episode_id),
              client.getLatestFeedbackIdByMetric(episode_id),
            ]);
            return { feedbacks, bounds, latestFeedbackByMetric };
          },
        )
      : Promise.all([
          client.getFeedbackByTargetId(episode_id, { limit: 10 }),
          client.getFeedbackBoundsByTargetId(episode_id),
          client.getLatestFeedbackIdByMetric(episode_id),
        ]).then(([feedbacks, bounds, latestFeedbackByMetric]) => ({
          feedbacks,
          bounds,
          latestFeedbackByMetric,
        }));

    const [inferencesData, numFeedbacks, feedbackData] = await Promise.all([
      inferencesPromise,
      numFeedbacksPromise,
      feedbackDataPromise,
    ]);

    const episodeData: EpisodeDetailData = {
      episode_id,
      inferences: inferencesData.inferences,
      num_inferences: Number(numInferences),
      num_feedbacks: numFeedbacks,
      feedback: feedbackData.feedbacks,
      feedback_bounds: feedbackData.bounds,
      latestFeedbackByMetric: feedbackData.latestFeedbackByMetric,
    };

    return Response.json(episodeData);
  } catch (error) {
    if (error instanceof Response) {
      throw error;
    }
    logger.error("Failed to fetch episode:", error);
    throw data("Failed to fetch episode details", { status: 500 });
  }
}
