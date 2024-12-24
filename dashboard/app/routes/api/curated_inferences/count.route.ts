import type { LoaderFunctionArgs } from "react-router";
import {
  countCuratedInferences,
  countFeedbacksForMetric,
  countInferencesForFunction,
} from "~/utils/clickhouse";
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
  // For type-safe fetching of counts, we would want this function to return a Promise<CountsData>
  const url = new URL(request.url);
  const functionName = url.searchParams.get("function");
  const metricName = url.searchParams.get("metric");
  const threshold = parseFloat(url.searchParams.get("threshold") || "0");

  let inferenceCount = null;
  let feedbackCount = null;
  let curatedInferenceCount = null;
  const config = await getConfig();
  if (functionName) {
    inferenceCount = await countInferencesForFunction(
      functionName,
      config.functions[functionName],
    );
  }
  if (metricName) {
    feedbackCount = await countFeedbacksForMetric(
      metricName,
      config.metrics[metricName],
    );
  }
  if (functionName && metricName) {
    curatedInferenceCount = await countCuratedInferences(
      functionName,
      config.functions[functionName],
      metricName,
      config.metrics[metricName],
      threshold,
    );
  }
  // For type-safe fetching of counts, we would want this return statement to be:
  // return {
  //   inferenceCount,
  //   feedbackCount,
  //   curatedInferenceCount,
  // };
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
