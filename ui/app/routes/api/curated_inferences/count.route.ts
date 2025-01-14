import type { LoaderFunctionArgs } from "react-router";
import {
  countCuratedInferences,
  countFeedbacksForMetric,
} from "~/utils/clickhouse/curation";
import { countInferencesForFunction } from "~/utils/clickhouse/inference";
import { getConfig } from "~/utils/config/index.server";

/// Count the number of inferences, feedbacks, and curated inferences for a given function and metric
/// This is used to determine the number of inferences to display in the UI
/// Call this route with optional function and metric parameters to get the counts
/// If only a function is provided, it will count all inferences for that function
/// If only a metric is provided, it will count all feedbacks for that metric
/// If both a function and metric are provided, it will count all curated inferences for that function and metric
export async function loader({
  request,
}: LoaderFunctionArgs): Promise<Response> {
  const url = new URL(request.url);
  const functionName = url.searchParams.get("function");
  const metricName = url.searchParams.get("metric");
  const threshold = parseFloat(url.searchParams.get("threshold") || "0");

  const config = await getConfig();

  // Run all fetches concurrently
  const [inferenceCount, feedbackCount, curatedInferenceCount] =
    await Promise.all([
      functionName
        ? countInferencesForFunction(
            functionName,
            config.functions[functionName],
          )
        : Promise.resolve(null),

      functionName && metricName
        ? countFeedbacksForMetric(
            functionName,
            config.functions[functionName],
            metricName,
            config.metrics[metricName],
          )
        : Promise.resolve(null),

      functionName && metricName
        ? countCuratedInferences(
            functionName,
            config.functions[functionName],
            metricName,
            config.metrics[metricName],
            threshold,
          )
        : Promise.resolve(null),
    ]);

  return Response.json({
    inferenceCount,
    feedbackCount,
    curatedInferenceCount,
  } as CountsData);
}

export interface CountsData {
  inferenceCount: number | null;
  feedbackCount: number | null;
  curatedInferenceCount: number | null;
}
