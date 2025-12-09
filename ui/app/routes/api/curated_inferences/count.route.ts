import { useEffect } from "react";
import { data, useFetcher, type LoaderFunctionArgs } from "react-router";
import {
  countCuratedInferences,
  countFeedbacksForMetric,
} from "~/utils/clickhouse/curation.server";
import { countInferencesForFunction } from "~/utils/clickhouse/inference.server";
import { getFeedbackConfig } from "~/utils/config/feedback";
import { getConfig, getFunctionConfig } from "~/utils/config/index.server";

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
  const functionConfig = functionName
    ? await getFunctionConfig(functionName, config)
    : null;
  if (functionName && !functionConfig) {
    throw data(`Function ${functionName} not found in config`, { status: 404 });
  }
  const metricConfig = getFeedbackConfig(metricName || "", config);
  if (metricName && !metricConfig) {
    throw data(`Metric ${metricName} not found in config`, { status: 404 });
  }

  // Run all fetches concurrently
  const [inferenceCount, feedbackCount, curatedInferenceCount] =
    await Promise.all([
      functionName ? countInferencesForFunction(functionName) : null,

      functionName && functionConfig && metricName && metricConfig
        ? countFeedbacksForMetric(
            functionName,
            functionConfig,
            metricName,
            metricConfig,
          )
        : Promise.resolve(null),

      functionName && functionConfig && metricName && metricConfig
        ? countCuratedInferences(
            functionName,
            functionConfig,
            metricName,
            metricConfig,
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

/**
 * A hook that fetches counts for inferences, feedbacks, and curated inferences based on function, metric, and threshold parameters.
 * This hook automatically refetches when any of the parameters change.
 *
 * @param params.functionName - The name of the function to get counts for
 * @param params.metricName - Optional metric name to filter counts by
 * @param params.threshold - Optional threshold value for curated inferences
 * @returns An object containing:
 *  - inferenceCount: Total number of inferences for the function
 *  - feedbackCount: Number of feedbacks for the function/metric combination
 *  - curatedInferenceCount: Number of curated inferences meeting the threshold criteria
 *  - isLoading: Whether the counts are currently being fetched
 */
export function useCountFetcher(params: {
  functionName?: string;
  metricName?: string;
  threshold?: number;
}): CountsData & { isLoading: boolean } {
  const countFetcher = useFetcher();

  useEffect(() => {
    if (params.functionName) {
      const searchParams = new URLSearchParams();
      searchParams.set("function", params.functionName);
      if (params.metricName) searchParams.set("metric", params.metricName);
      if (params.threshold)
        searchParams.set("threshold", String(params.threshold));

      countFetcher.load(`/api/curated_inferences/count?${searchParams}`);
    }
    // TODO: Fix and stop ignoring lint rule
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params.functionName, params.metricName, params.threshold]);

  return {
    inferenceCount: countFetcher.data?.inferenceCount ?? null,
    feedbackCount: countFetcher.data?.feedbackCount ?? null,
    curatedInferenceCount: countFetcher.data?.curatedInferenceCount ?? null,
    isLoading: countFetcher.state === "loading",
  };
}
