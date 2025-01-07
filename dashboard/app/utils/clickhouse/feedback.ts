import { z } from "zod";
import type { TableBounds } from "./common";
import { data } from "react-router";
import { clickhouseClient } from "./common";

export const booleanMetricFeedbackRowSchema = z.object({
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
          id,
          target_id,
          metric_name,
          value,
          tags,
          UUIDv7ToDateTime(id) AS timestamp
        FROM BooleanMetricFeedbackByTargetId
        WHERE target_id = {target_id:String}
        ORDER BY toUInt128(id) DESC
        LIMIT {page_size:UInt32}
      `;
  } else if (before) {
    query = `
        SELECT
          id,
          target_id,
          metric_name,
          value,
          tags,
          UUIDv7ToDateTime(id) AS timestamp
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
            UUIDv7ToDateTime(id) AS timestamp
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
    return rows;
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
      return { first_id: undefined, last_id: undefined };
    }
    return rows[0];
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
          id,
          target_id,
          target_type,
          value,
          UUIDv7ToDateTime(id) AS timestamp
        FROM CommentFeedbackByTargetId
        WHERE target_id = {target_id:String}
        ORDER BY toUInt128(id) DESC
        LIMIT {page_size:UInt32}
      `;
  } else if (before) {
    query = `
        SELECT
          id,
          target_id,
          target_type,
          value,
          UUIDv7ToDateTime(id) AS timestamp
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
            UUIDv7ToDateTime(id) AS timestamp
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
    return rows;
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
      return { first_id: undefined, last_id: undefined };
    }
    return rows[0];
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
          id,
          inference_id,
          value,
          UUIDv7ToDateTime(id) AS timestamp
        FROM DemonstrationFeedbackByInferenceId
        WHERE inference_id = {inference_id:String}
        ORDER BY toUInt128(id) DESC
        LIMIT {page_size:UInt32}
      `;
  } else if (before) {
    query = `
        SELECT
          id,
          inference_id,
          value,
          UUIDv7ToDateTime(id) AS timestamp
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
          id,
          inference_id,
          value,
          timestamp
        FROM
        (
          SELECT
            id,
            inference_id,
            value,
            UUIDv7ToDateTime(id) AS timestamp
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
    return rows;
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
      return { first_id: undefined, last_id: undefined };
    }
    return rows[0];
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
          id,
          target_id,
          metric_name,
          value,
          tags,
          UUIDv7ToDateTime(id) AS timestamp
        FROM FloatMetricFeedbackByTargetId
        WHERE target_id = {target_id:String}
        ORDER BY toUInt128(id) DESC
        LIMIT {page_size:UInt32}
      `;
  } else if (before) {
    query = `
        SELECT
          id,
          target_id,
          metric_name,
          value,
          tags,
          UUIDv7ToDateTime(id) AS timestamp
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
            UUIDv7ToDateTime(id) AS timestamp
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
    return rows;
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
      return { first_id: undefined, last_id: undefined };
    }
    return rows[0];
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

export const feedbackRowSchema = z.union([
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

  // Sort by id (which is a UUID v7 timestamp)
  allFeedback.sort((a, b) => a.id.localeCompare(b.id));

  // Take either earliest or latest elements based on pagination params
  if (before) {
    // If 'before' is specified, take latest elements
    return allFeedback.slice(0, page_size || 100);
  } else {
    // If 'after' is specified or no pagination params, take earliest elements
    return allFeedback.slice(-Math.min(allFeedback.length, page_size || 100));
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
