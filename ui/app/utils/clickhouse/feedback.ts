import { type TableBounds, TableBoundsSchema } from "./common";
import { data } from "react-router";
import { InferenceJoinKey } from "./common";
import { clickhouseClient } from "./client.server";
import type { MetricConfig } from "~/utils/config/metric";
import { getInferenceJoinKey } from "~/utils/clickhouse/curation";
import { z } from "zod";

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

export async function queryBooleanMetricsByTargetId(params: {
  target_id: string;
  before?: string;
  after?: string;
  page_size?: number;
}): Promise<BooleanMetricFeedbackRow[]> {
  const { target_id, before, after, page_size } = params;

  if (before && after) {
    throw new Error("Cannot specify both 'before' and 'after' parameters");
  }

  let query = "";
  const query_params: Record<string, string | number> = {
    target_id,
    page_size: page_size || 100,
  };

  if (!before && !after) {
    query = `
        SELECT
          'boolean' AS type,
          id,
          target_id,
          metric_name,
          value,
          tags,
          formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
        FROM BooleanMetricFeedbackByTargetId
        WHERE target_id = {target_id:String}
        ORDER BY toUInt128(id) DESC
        LIMIT {page_size:UInt32}
      `;
  } else if (before) {
    query = `
        SELECT
          'boolean' AS type,
          id,
          target_id,
          metric_name,
          value,
          tags,
          formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
        FROM BooleanMetricFeedbackByTargetId
        WHERE target_id = {target_id:String}
          AND toUInt128(id) < toUInt128(toUUID({before:String}))
        ORDER BY toUInt128(id) DESC
        LIMIT {page_size:UInt32}
      `;
    query_params.before = before;
  } else if (after) {
    query = `
        SELECT
          'boolean' AS type,
          id,
          target_id,
          metric_name,
          value,
          tags,
          timestamp
        FROM
        (
          SELECT
            id,
            target_id,
            metric_name,
            value,
            tags,
            formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
          FROM BooleanMetricFeedbackByTargetId
          WHERE target_id = {target_id:String}
            AND toUInt128(id) > toUInt128(toUUID({after:String}))
          ORDER BY toUInt128(id) ASC
          LIMIT {page_size:UInt32}
        )
        ORDER BY toUInt128(id) DESC
      `;
    query_params.after = after;
  }

  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
      query_params,
    });
    const rows = await resultSet.json<BooleanMetricFeedbackRow>();
    return z.array(booleanMetricFeedbackRowSchema).parse(rows);
  } catch (error) {
    console.error(error);
    throw data("Error querying boolean metrics", { status: 500 });
  }
}

export async function queryBooleanMetricFeedbackBoundsByTargetId(params: {
  target_id: string;
}): Promise<TableBounds> {
  const { target_id } = params;
  const query = `
     SELECT
    (SELECT id FROM BooleanMetricFeedbackByTargetId WHERE toUInt128(id) = (SELECT MIN(toUInt128(id)) FROM BooleanMetricFeedbackByTargetId WHERE target_id = {target_id:String})) AS first_id,
    (SELECT id FROM BooleanMetricFeedbackByTargetId WHERE toUInt128(id) = (SELECT MAX(toUInt128(id)) FROM BooleanMetricFeedbackByTargetId WHERE target_id = {target_id:String})) AS last_id
    FROM BooleanMetricFeedbackByTargetId
    LIMIT 1
    `;

  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
      query_params: { target_id },
    });
    const rows = await resultSet.json<TableBounds>();
    // If there is no data at all ClickHouse returns an empty array
    // If there is no data for a specific target_id ClickHouse returns an array with a single element where first_id and last_id are null
    // We handle both cases by returning undefined for first_id and last_id
    if (
      rows.length === 0 ||
      (rows[0].first_id === null && rows[0].last_id === null)
    ) {
      return { first_id: null, last_id: null };
    }
    return TableBoundsSchema.parse(rows[0]);
  } catch (error) {
    console.error(error);
    throw data("Error querying boolean metric feedback bounds", {
      status: 500,
    });
  }
}

export async function countBooleanMetricFeedbackByTargetId(
  target_id: string,
): Promise<number> {
  const query = `SELECT COUNT() AS count FROM BooleanMetricFeedbackByTargetId WHERE target_id = {target_id:String}`;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { target_id },
  });
  const rows = await resultSet.json<{ count: string }>();
  return Number(rows[0].count);
}

export const commentFeedbackRowSchema = z.object({
  type: z.literal("comment"),
  id: z.string().uuid(),
  target_id: z.string().uuid(),
  target_type: z.enum(["inference", "episode"]),
  value: z.string(),
  timestamp: z.string().datetime(),
});

export type CommentFeedbackRow = z.infer<typeof commentFeedbackRowSchema>;

export async function queryCommentFeedbackByTargetId(params: {
  target_id: string;
  before?: string;
  after?: string;
  page_size?: number;
}): Promise<CommentFeedbackRow[]> {
  const { target_id, before, after, page_size } = params;

  if (before && after) {
    throw new Error("Cannot specify both 'before' and 'after' parameters");
  }

  let query = "";
  const query_params: Record<string, string | number> = {
    target_id,
    page_size: page_size || 100,
  };

  if (!before && !after) {
    query = `
        SELECT
          'comment' AS type,
          id,
          target_id,
          target_type,
          value,
          formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
        FROM CommentFeedbackByTargetId
        WHERE target_id = {target_id:String}
        ORDER BY toUInt128(id) DESC
        LIMIT {page_size:UInt32}
      `;
  } else if (before) {
    query = `
        SELECT
          'comment' AS type,
          id,
          target_id,
          target_type,
          value,
          formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
        FROM CommentFeedbackByTargetId
        WHERE target_id = {target_id:String}
          AND toUInt128(id) < toUInt128(toUUID({before:String}))
        ORDER BY toUInt128(id) DESC
        LIMIT {page_size:UInt32}
      `;
    query_params.before = before;
  } else if (after) {
    query = `
        SELECT
          'comment' AS type,
          id,
          target_id,
          target_type,
          value,
          timestamp
        FROM
        (
          SELECT
            id,
            target_id,
            target_type,
            value,
            formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
          FROM CommentFeedbackByTargetId
          WHERE target_id = {target_id:String}
            AND toUInt128(id) > toUInt128(toUUID({after:String}))
          ORDER BY toUInt128(id) ASC
          LIMIT {page_size:UInt32}
        )
        ORDER BY toUInt128(id) DESC
      `;
    query_params.after = after;
  }

  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
      query_params,
    });
    const rows = await resultSet.json<CommentFeedbackRow>();
    return z.array(commentFeedbackRowSchema).parse(rows);
  } catch (error) {
    console.error(error);
    throw data("Error querying comment feedback", { status: 500 });
  }
}

export async function queryCommentFeedbackBoundsByTargetId(params: {
  target_id: string;
}): Promise<TableBounds> {
  const { target_id } = params;
  const query = `
     SELECT
    (SELECT id FROM CommentFeedbackByTargetId WHERE toUInt128(id) = (SELECT MIN(toUInt128(id)) FROM CommentFeedbackByTargetId WHERE target_id = {target_id:String})) AS first_id,
    (SELECT id FROM CommentFeedbackByTargetId WHERE toUInt128(id) = (SELECT MAX(toUInt128(id)) FROM CommentFeedbackByTargetId WHERE target_id = {target_id:String})) AS last_id
    FROM CommentFeedbackByTargetId
    LIMIT 1
    `;

  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
      query_params: { target_id },
    });
    const rows = await resultSet.json<TableBounds>();
    // If there is no data at all ClickHouse returns an empty array
    // If there is no data for a specific target_id ClickHouse returns an array with a single element where first_id and last_id are null
    // We handle both cases by returning undefined for first_id and last_id
    if (
      rows.length === 0 ||
      (rows[0].first_id === null && rows[0].last_id === null)
    ) {
      return { first_id: null, last_id: null };
    }
    return TableBoundsSchema.parse(rows[0]);
  } catch (error) {
    console.error(error);
    throw data("Error querying comment feedback bounds", { status: 500 });
  }
}

export async function countCommentFeedbackByTargetId(
  target_id: string,
): Promise<number> {
  const query = `SELECT COUNT() AS count FROM CommentFeedbackByTargetId WHERE target_id = {target_id:String}`;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { target_id },
  });
  const rows = await resultSet.json<{ count: string }>();
  return Number(rows[0].count);
}

export const demonstrationFeedbackRowSchema = z.object({
  type: z.literal("demonstration"),
  id: z.string().uuid(),
  inference_id: z.string().uuid(),
  value: z.string(),
  timestamp: z.string().datetime(),
});

export type DemonstrationFeedbackRow = z.infer<
  typeof demonstrationFeedbackRowSchema
>;

export async function queryDemonstrationFeedbackByInferenceId(params: {
  inference_id: string;
  before?: string;
  after?: string;
  page_size?: number;
}): Promise<DemonstrationFeedbackRow[]> {
  const { inference_id, before, after, page_size } = params;

  if (before && after) {
    throw new Error("Cannot specify both 'before' and 'after' parameters");
  }

  let query = "";
  const query_params: Record<string, string | number> = {
    inference_id,
    page_size: page_size || 100,
  };

  if (!before && !after) {
    query = `
        SELECT
          'demonstration' AS type,
          id,
          inference_id,
          value,
          formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
        FROM DemonstrationFeedbackByInferenceId
        WHERE inference_id = {inference_id:String}
        ORDER BY toUInt128(id) DESC
        LIMIT {page_size:UInt32}
      `;
  } else if (before) {
    query = `
        SELECT
          'demonstration' AS type,
          id,
          inference_id,
          value,
          formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
        FROM DemonstrationFeedbackByInferenceId
        WHERE inference_id = {inference_id:String}
          AND toUInt128(id) < toUInt128(toUUID({before:String}))
        ORDER BY toUInt128(id) DESC
        LIMIT {page_size:UInt32}
      `;
    query_params.before = before;
  } else if (after) {
    query = `
        SELECT
          'demonstration' AS type,
          id,
          inference_id,
          value,
          formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
        FROM
        (
          SELECT
            id,
            inference_id,
            value,
            formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
          FROM DemonstrationFeedbackByInferenceId
          WHERE inference_id = {inference_id:String}
            AND toUInt128(id) > toUInt128(toUUID({after:String}))
          ORDER BY toUInt128(id) ASC
          LIMIT {page_size:UInt32}
        )
        ORDER BY toUInt128(id) DESC
      `;
    query_params.after = after;
  }

  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
      query_params,
    });
    const rows = await resultSet.json<DemonstrationFeedbackRow>();
    return z.array(demonstrationFeedbackRowSchema).parse(rows);
  } catch (error) {
    console.error(error);
    throw data("Error querying demonstration feedback", { status: 500 });
  }
}

export async function queryDemonstrationFeedbackBoundsByInferenceId(params: {
  inference_id: string;
}): Promise<TableBounds> {
  const { inference_id } = params;
  const query = `
     SELECT
    (SELECT id FROM DemonstrationFeedbackByInferenceId WHERE toUInt128(id) = (SELECT MIN(toUInt128(id)) FROM DemonstrationFeedbackByInferenceId WHERE inference_id = {inference_id:String})) AS first_id,
    (SELECT id FROM DemonstrationFeedbackByInferenceId WHERE toUInt128(id) = (SELECT MAX(toUInt128(id)) FROM DemonstrationFeedbackByInferenceId WHERE inference_id = {inference_id:String})) AS last_id
    FROM DemonstrationFeedbackByInferenceId
    LIMIT 1
    `;

  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
      query_params: { inference_id },
    });
    const rows = await resultSet.json<TableBounds>();
    // If there is no data at all ClickHouse returns an empty array
    // If there is no data for a specific target_id ClickHouse returns an array with a single element where first_id and last_id are null
    // We handle both cases by returning undefined for first_id and last_id
    if (
      rows.length === 0 ||
      (rows[0].first_id === null && rows[0].last_id === null)
    ) {
      return { first_id: null, last_id: null };
    }
    return TableBoundsSchema.parse(rows[0]);
  } catch (error) {
    console.error(error);
    throw data("Error querying demonstration feedback bounds", {
      status: 500,
    });
  }
}

export async function countDemonstrationFeedbackByInferenceId(
  inference_id: string,
): Promise<number> {
  const query = `SELECT COUNT() AS count FROM DemonstrationFeedbackByInferenceId WHERE inference_id = {inference_id:String}`;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { inference_id },
  });
  const rows = await resultSet.json<{ count: string }>();
  return Number(rows[0].count);
}

export const floatMetricFeedbackRowSchema = z
  .object({
    type: z.literal("float"),
    id: z.string().uuid(),
    target_id: z.string().uuid(),
    metric_name: z.string(),
    value: z.number(),
    tags: z.record(z.string(), z.string()),
    timestamp: z.string().datetime(),
  })
  .strict();

export type FloatMetricFeedbackRow = z.infer<
  typeof floatMetricFeedbackRowSchema
>;

export async function queryFloatMetricsByTargetId(params: {
  target_id: string;
  before?: string;
  after?: string;
  page_size?: number;
}): Promise<FloatMetricFeedbackRow[]> {
  const { target_id, before, after, page_size } = params;

  if (before && after) {
    throw new Error("Cannot specify both 'before' and 'after' parameters");
  }

  let query = "";
  const query_params: Record<string, string | number> = {
    target_id,
    page_size: page_size || 100,
  };

  if (!before && !after) {
    query = `
        SELECT
          'float' AS type,
          id,
          target_id,
          metric_name,
          value,
          tags,
          formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
        FROM FloatMetricFeedbackByTargetId
        WHERE target_id = {target_id:String}
        ORDER BY toUInt128(id) DESC
        LIMIT {page_size:UInt32}
      `;
  } else if (before) {
    query = `
        SELECT
          'float' AS type,
          id,
          target_id,
          metric_name,
          value,
          tags,
          formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
        FROM FloatMetricFeedbackByTargetId
        WHERE target_id = {target_id:String}
          AND toUInt128(id) < toUInt128(toUUID({before:String}))
        ORDER BY toUInt128(id) DESC
        LIMIT {page_size:UInt32}
      `;
    query_params.before = before;
  } else if (after) {
    query = `
        SELECT
          'float' AS type,
          id,
          target_id,
          metric_name,
          value,
          tags,
          timestamp
        FROM
        (
          SELECT
            id,
            target_id,
            metric_name,
            value,
            tags,
            formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
          FROM FloatMetricFeedbackByTargetId
          WHERE target_id = {target_id:String}
            AND toUInt128(id) > toUInt128(toUUID({after:String}))
          ORDER BY toUInt128(id) ASC
          LIMIT {page_size:UInt32}
        )
        ORDER BY toUInt128(id) DESC
      `;
    query_params.after = after;
  }

  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
      query_params,
    });
    const rows = await resultSet.json<FloatMetricFeedbackRow>();
    return z.array(floatMetricFeedbackRowSchema).parse(rows);
  } catch (error) {
    console.error(error);
    throw data("Error querying float metric feedback", { status: 500 });
  }
}

export async function queryFloatMetricFeedbackBoundsByTargetId(params: {
  target_id: string;
}): Promise<TableBounds> {
  const { target_id } = params;
  const query = `
     SELECT
    (SELECT id FROM FloatMetricFeedbackByTargetId WHERE toUInt128(id) = (SELECT MIN(toUInt128(id)) FROM FloatMetricFeedbackByTargetId WHERE target_id = {target_id:String})) AS first_id,
    (SELECT id FROM FloatMetricFeedbackByTargetId WHERE toUInt128(id) = (SELECT MAX(toUInt128(id)) FROM FloatMetricFeedbackByTargetId WHERE target_id = {target_id:String})) AS last_id
    FROM FloatMetricFeedbackByTargetId
    LIMIT 1
    `;

  try {
    const resultSet = await clickhouseClient.query({
      query,
      format: "JSONEachRow",
      query_params: { target_id },
    });
    const rows = await resultSet.json<TableBounds>();
    // If there is no data at all ClickHouse returns an empty array
    // If there is no data for a specific target_id ClickHouse returns an array with a single element where first_id and last_id are null
    // We handle both cases by returning undefined for first_id and last_id
    if (
      rows.length === 0 ||
      (rows[0].first_id === null && rows[0].last_id === null)
    ) {
      return { first_id: null, last_id: null };
    }
    return TableBoundsSchema.parse(rows[0]);
  } catch (error) {
    console.error(error);
    throw data("Error querying float metric feedback bounds", {
      status: 500,
    });
  }
}

export async function countFloatMetricFeedbackByTargetId(
  target_id: string,
): Promise<number> {
  const query = `SELECT COUNT() AS count FROM FloatMetricFeedbackByTargetId WHERE target_id = {target_id:String}`;
  const resultSet = await clickhouseClient.query({
    query,
    format: "JSONEachRow",
    query_params: { target_id },
  });
  const rows = await resultSet.json<{ count: string }>();
  return Number(rows[0].count);
}
export const feedbackRowSchema = z.discriminatedUnion("type", [
  booleanMetricFeedbackRowSchema,
  floatMetricFeedbackRowSchema,
  commentFeedbackRowSchema,
  demonstrationFeedbackRowSchema,
]);

export type FeedbackRow = z.infer<typeof feedbackRowSchema>;

export async function queryFeedbackByTargetId(params: {
  target_id: string;
  before?: string;
  after?: string;
  page_size?: number;
}): Promise<FeedbackRow[]> {
  const { target_id, before, after, page_size } = params;

  const [booleanMetrics, commentFeedback, demonstrationFeedback, floatMetrics] =
    await Promise.all([
      queryBooleanMetricsByTargetId({
        target_id,
        before,
        after,
        page_size,
      }),
      queryCommentFeedbackByTargetId({
        target_id,
        before,
        after,
        page_size,
      }),
      queryDemonstrationFeedbackByInferenceId({
        inference_id: target_id,
        before,
        after,
        page_size,
      }),
      queryFloatMetricsByTargetId({
        target_id,
        before,
        after,
        page_size,
      }),
    ]);

  // Combine all feedback types into a single array
  const allFeedback: FeedbackRow[] = [
    ...booleanMetrics,
    ...commentFeedback,
    ...demonstrationFeedback,
    ...floatMetrics,
  ];

  // Sort by id (which is a UUID v7 timestamp) in descending order
  allFeedback.sort((a, b) => (b.id > a.id ? 1 : -1));

  // Take either earliest or latest elements based on pagination params
  if (after) {
    // If 'after' is specified, take earliest elements
    return allFeedback.slice(-Math.min(allFeedback.length, page_size || 100));
  } else {
    // If 'before' is specified or no pagination params, take latest elements
    return allFeedback.slice(0, page_size || 100);
  }
}

export async function queryFeedbackBoundsByTargetId(params: {
  target_id: string;
}): Promise<TableBounds> {
  const { target_id } = params;
  const [
    booleanMetricFeedbackBounds,
    commentFeedbackBounds,
    demonstrationFeedbackBounds,
    floatMetricFeedbackBounds,
  ] = await Promise.all([
    queryBooleanMetricFeedbackBoundsByTargetId({
      target_id,
    }),
    queryCommentFeedbackBoundsByTargetId({
      target_id,
    }),
    queryDemonstrationFeedbackBoundsByInferenceId({
      inference_id: target_id,
    }),
    queryFloatMetricFeedbackBoundsByTargetId({
      target_id,
    }),
  ]);

  // Find the earliest first_id and latest last_id across all feedback types
  const allFirstIds = [
    booleanMetricFeedbackBounds.first_id,
    commentFeedbackBounds.first_id,
    demonstrationFeedbackBounds.first_id,
    floatMetricFeedbackBounds.first_id,
  ].filter((id): id is string => id !== undefined);

  const allLastIds = [
    booleanMetricFeedbackBounds.last_id,
    commentFeedbackBounds.last_id,
    demonstrationFeedbackBounds.last_id,
    floatMetricFeedbackBounds.last_id,
  ].filter((id): id is string => id !== undefined);

  return {
    first_id: allFirstIds.sort()[0],
    last_id: allLastIds.sort().reverse()[0],
  };
}

export async function countFeedbackByTargetId(
  target_id: string,
): Promise<number> {
  const [booleanMetrics, commentFeedback, demonstrationFeedback, floatMetrics] =
    await Promise.all([
      countBooleanMetricFeedbackByTargetId(target_id),
      countCommentFeedbackByTargetId(target_id),
      countDemonstrationFeedbackByInferenceId(target_id),
      countFloatMetricFeedbackByTargetId(target_id),
    ]);
  return (
    booleanMetrics + commentFeedback + demonstrationFeedback + floatMetrics
  );
}

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
  metrics: Record<string, MetricConfig>;
  variant_name?: string;
}): Promise<MetricsWithFeedbackData> {
  const { function_name, inference_table, metrics, variant_name } = params;

  const inferenceMetrics = Object.entries(metrics)
    .filter(([, metric]) => {
      try {
        return (
          "level" in metric &&
          getInferenceJoinKey(metric.level) === InferenceJoinKey.ID
        );
      } catch {
        return false;
      }
    })
    .map(([name]) => name);

  const episodeMetrics = Object.entries(metrics)
    .filter(([, metric]) => {
      try {
        return (
          "level" in metric &&
          getInferenceJoinKey(metric.level) === InferenceJoinKey.EPISODE_ID
        );
      } catch {
        return false;
      }
    })
    .map(([name]) => name);

  const idInClause =
    inferenceMetrics.length > 0
      ? `IN ('${inferenceMetrics.join("','")}')`
      : `= ''`;

  const episodeIdInClause =
    episodeMetrics.length > 0 ? `IN ('${episodeMetrics.join("','")}')` : `= ''`;

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
        AND bmf.metric_name ${idInClause}
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
        AND bmf.metric_name ${episodeIdInClause}
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
        AND fmf.metric_name ${idInClause}
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
        AND fmf.metric_name ${episodeIdInClause}
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
    const resultSet = await clickhouseClient.query({
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
    console.error("Error fetching metrics with feedback:", error);
    throw data("Error fetching metrics with feedback", { status: 500 });
  }
}
