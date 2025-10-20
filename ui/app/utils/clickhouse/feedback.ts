import { data } from "react-router";
import { getClickhouseClient } from "./client.server";
import { getNativeDatabaseClient } from "../tensorzero/native_client.server";
import { z } from "zod";
import { logger } from "~/utils/logger";
import type { FeedbackRow } from "tensorzero-node";

export const booleanMetricFeedbackRowSchema = z.object({
  type: z.literal("boolean"),
  id: z.string().uuid(),
  target_id: z.string().uuid(),
  metric_name: z.string(),
  value: z.boolean(),
  tags: z.record(z.string(), z.string()),
  timestamp: z.string().datetime(),
});

export type BooleanMetricFeedbackRow = z.infer<
  typeof booleanMetricFeedbackRowSchema
>;

export const commentFeedbackRowSchema = z.object({
  type: z.literal("comment"),
  id: z.string().uuid(),
  target_id: z.string().uuid(),
  target_type: z.enum(["inference", "episode"]),
  value: z.string(),
  timestamp: z.string().datetime(),
  tags: z.record(z.string(), z.string()),
});

export const metricsWithFeedbackRowSchema = z
  .object({
    function_name: z.string(),
    metric_name: z.string(),
    metric_type: z.enum(["boolean", "float", "demonstration"]),
    feedback_count: z.number(),
  })
  .strict();

export const metricsWithFeedbackDataSchema = z
  .object({
    metrics: z.array(metricsWithFeedbackRowSchema),
  })
  .strict();

export type MetricsWithFeedbackRow = z.infer<
  typeof metricsWithFeedbackRowSchema
>;
export type MetricsWithFeedbackData = z.infer<
  typeof metricsWithFeedbackDataSchema
>;

export async function queryMetricsWithFeedback(params: {
  function_name: string;
  inference_table: string;
  variant_name?: string;
}): Promise<MetricsWithFeedbackData> {
  const { function_name, inference_table, variant_name } = params;

  const variantClause = variant_name
    ? `AND i.variant_name = {variant_name:String}`
    : "";

  const query = `
    WITH
    boolean_inference_metrics AS (
      SELECT
        i.function_name,
        bmf.metric_name,
        'boolean' as metric_type,
        COUNT(DISTINCT i.id) as feedback_count
      FROM ${inference_table} i
      JOIN BooleanMetricFeedback bmf ON bmf.target_id = i.id
      WHERE i.function_name = {function_name:String}
        ${variantClause}
      GROUP BY i.function_name, bmf.metric_name
      HAVING feedback_count > 0
    ),

    boolean_episode_metrics AS (
      SELECT
        i.function_name,
        bmf.metric_name,
        'boolean' as metric_type,
        COUNT(DISTINCT i.id) as feedback_count
      FROM ${inference_table} i
      JOIN BooleanMetricFeedback bmf ON bmf.target_id = i.episode_id
      WHERE i.function_name = {function_name:String}
        ${variantClause}
      GROUP BY i.function_name, bmf.metric_name
      HAVING feedback_count > 0
    ),

    float_inference_metrics AS (
      SELECT
        i.function_name,
        fmf.metric_name,
        'float' as metric_type,
        COUNT(DISTINCT i.id) as feedback_count
      FROM ${inference_table} i
      JOIN FloatMetricFeedback fmf ON fmf.target_id = i.id
      WHERE i.function_name = {function_name:String}
        ${variantClause}
      GROUP BY i.function_name, fmf.metric_name
      HAVING feedback_count > 0
    ),

    float_episode_metrics AS (
      SELECT
        i.function_name,
        fmf.metric_name,
        'float' as metric_type,
        COUNT(DISTINCT i.id) as feedback_count
      FROM ${inference_table} i
      JOIN FloatMetricFeedback fmf ON fmf.target_id = i.episode_id
      WHERE i.function_name = {function_name:String}
        ${variantClause}
      GROUP BY i.function_name, fmf.metric_name
      HAVING feedback_count > 0
    ),
    demonstration_metrics AS (
      SELECT
        i.function_name,
        'demonstration' as metric_name,
        'demonstration' as metric_type,
        COUNT(DISTINCT i.id) as feedback_count
      FROM ${inference_table} i
      JOIN DemonstrationFeedback df ON df.inference_id = i.id
      WHERE i.function_name = {function_name:String}
        ${variantClause}
      GROUP BY i.function_name
      HAVING feedback_count > 0
    )
    SELECT
      function_name,
      metric_name,
      metric_type,
      toString(feedback_count) as feedback_count
    FROM (
      SELECT * FROM boolean_inference_metrics
      UNION ALL
      SELECT * FROM boolean_episode_metrics
      UNION ALL
      SELECT * FROM float_inference_metrics
      UNION ALL
      SELECT * FROM float_episode_metrics
      UNION ALL
      SELECT * FROM demonstration_metrics
    )
    ORDER BY metric_type, metric_name`;

  try {
    const resultSet = await getClickhouseClient().query({
      query,
      format: "JSONEachRow",
      query_params: {
        function_name,
        ...(variant_name && { variant_name }),
      },
    });

    const rawMetrics = (await resultSet.json()) as Array<{
      function_name: string;
      metric_name: string;
      metric_type: "boolean" | "float" | "demonstration";
      feedback_count: string;
    }>;

    const validMetrics = rawMetrics.map((metric) => ({
      ...metric,
      feedback_count: Number(metric.feedback_count),
    }));

    return metricsWithFeedbackDataSchema.parse({ metrics: validMetrics });
  } catch (error) {
    logger.error("Error fetching metrics with feedback:", error);
    throw data("Error fetching metrics with feedback", { status: 500 });
  }
}

/**
 * Polls for a specific feedback item on the first page.
 * @param targetId The ID of the target (e.g., inference_id).
 * @param feedbackId The ID of the feedback item to find.
 * @param pageSize The number of items per page to fetch.
 * @param maxRetries Maximum number of polling attempts.
 * @param retryDelay Delay between retries in milliseconds.
 * @returns The fetched feedback list.
 */
export async function pollForFeedbackItem(
  targetId: string,
  feedbackId: string,
  pageSize: number,
  maxRetries: number = 10,
  retryDelay: number = 200,
): Promise<FeedbackRow[]> {
  const dbClient = await getNativeDatabaseClient();

  let feedback: FeedbackRow[] = [];
  let found = false;
  for (let i = 0; i < maxRetries; i++) {
    feedback = await dbClient.queryFeedbackByTargetId({
      target_id: targetId,
      before: undefined,
      after: undefined,
      page_size: pageSize,
    });
    if (feedback.some((f) => f.id === feedbackId)) {
      found = true;
      break;
    }
    if (i < maxRetries - 1) {
      // Don't sleep after the last attempt
      await new Promise((resolve) => setTimeout(resolve, retryDelay));
    }
  }
  if (!found) {
    logger.warn(
      `Feedback ${feedbackId} for target ${targetId} not found after ${maxRetries} retries.`,
    );
  }
  return feedback;
}

export async function queryLatestFeedbackIdByMetric(params: {
  target_id: string;
}): Promise<Record<string, string>> {
  const { target_id } = params;

  const query = `
    SELECT
      metric_name,
      argMax(id, toUInt128(id)) as latest_id
    FROM BooleanMetricFeedbackByTargetId
    WHERE target_id = {target_id:String}
    GROUP BY metric_name

    UNION ALL

    SELECT
      metric_name,
      argMax(id, toUInt128(id)) as latest_id
    FROM FloatMetricFeedbackByTargetId
    WHERE target_id = {target_id:String}
    GROUP BY metric_name

    ORDER BY metric_name
  `;

  try {
    const resultSet = await getClickhouseClient().query({
      query,
      format: "JSONEachRow",
      query_params: { target_id },
    });
    const rows = await resultSet.json();

    const latestFeedbackByMetric = z
      .array(
        z.object({
          metric_name: z.string(),
          latest_id: z.string().uuid(),
        }),
      )
      .parse(rows);

    return Object.fromEntries(
      latestFeedbackByMetric.map((item) => [item.metric_name, item.latest_id]),
    );
  } catch (error) {
    logger.error("ERROR", error);
    throw data("Error querying latest feedback by metric", { status: 500 });
  }
}
