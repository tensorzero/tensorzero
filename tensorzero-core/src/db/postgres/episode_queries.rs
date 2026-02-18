//! EpisodeQueries implementation for Postgres.
//!
//! This module implements episode queries for the Postgres database.

use async_trait::async_trait;
use sqlx::{PgPool, QueryBuilder};
use uuid::Uuid;

use crate::config::Config;
use crate::db::{EpisodeByIdRow, EpisodeQueries, TableBoundsWithCount};
use crate::endpoints::stored_inferences::v1::types::InferenceFilter;
use crate::error::{Error, ErrorDetails};

use super::PostgresConnectionInfo;
use super::inference_filter_helpers::apply_inference_filter;

#[async_trait]
impl EpisodeQueries for PostgresConnectionInfo {
    async fn query_episode_table(
        &self,
        config: &Config,
        limit: u32,
        before: Option<Uuid>,
        after: Option<Uuid>,
        function_name: Option<String>,
        filters: Option<InferenceFilter>,
    ) -> Result<Vec<EpisodeByIdRow>, Error> {
        let pool = self.get_pool_result()?;
        query_episode_table_impl(
            pool,
            config,
            limit,
            before,
            after,
            function_name.as_deref(),
            filters.as_ref(),
        )
        .await
    }

    async fn query_episode_table_bounds(&self) -> Result<TableBoundsWithCount, Error> {
        let pool = self.get_pool_result()?;
        query_episode_table_bounds_impl(pool).await
    }
}

// =====================================================================
// Query builder functions (for unit testing)
// =====================================================================

/// Result of building the episode table query, containing the query builder
/// and whether results need to be reversed (for "after" pagination).
struct EpisodeTableQueryResult {
    query_builder: QueryBuilder<sqlx::Postgres>,
    reverse_results: bool,
}

/// Builds a query to fetch episodes with pagination and optional filtering.
///
/// Returns an error if both `before` and `after` are specified, or if
/// the filter references an invalid metric.
fn build_query_episode_table(
    config: &Config,
    limit: u32,
    before: Option<Uuid>,
    after: Option<Uuid>,
    function_name: Option<&str>,
    filters: Option<&InferenceFilter>,
) -> Result<EpisodeTableQueryResult, Error> {
    if before.is_some() && after.is_some() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "Cannot specify both before and after in query_episode_table".to_string(),
        }));
    }

    let has_filters = function_name.is_some() || filters.is_some();

    if has_filters {
        build_filtered_episode_query(config, limit, before, after, function_name, filters)
    } else {
        Ok(build_unfiltered_episode_query(limit, before, after))
    }
}

/// Builds an optimized episode listing query for the unfiltered case.
///
/// Fetches `limit` distinct episode_ids from each inference table using the
/// episode_id index (one backward/forward Merge Append per table), then
/// deduplicates and aggregates only those candidates. This avoids a full
/// table scan while keeping the query plan simple and parallelizable.
fn build_unfiltered_episode_query(
    limit: u32,
    before: Option<Uuid>,
    after: Option<Uuid>,
) -> EpisodeTableQueryResult {
    let reverse_results = after.is_some();
    let order = if after.is_some() { "ASC" } else { "DESC" };

    let mut qb = QueryBuilder::<sqlx::Postgres>::new("");

    // CTE 1: Get candidate episode_ids from both tables.
    // Each inner query uses the episode_id index with LIMIT for an efficient
    // single-pass Merge Append across partitions. The outer DISTINCT deduplicates
    // episodes that appear in both tables.
    qb.push(
        r"
        WITH candidate_episode_ids AS (
            SELECT DISTINCT episode_id FROM (
                (SELECT DISTINCT episode_id FROM tensorzero.chat_inferences",
    );

    if let Some(before_id) = before {
        qb.push(" WHERE episode_id < ");
        qb.push_bind(before_id);
    } else if let Some(after_id) = after {
        qb.push(" WHERE episode_id > ");
        qb.push_bind(after_id);
    }

    qb.push(format!(" ORDER BY episode_id {order} LIMIT "));
    qb.push_bind(limit as i64);

    qb.push(
        r")
                UNION ALL
                (SELECT DISTINCT episode_id FROM tensorzero.json_inferences",
    );

    if let Some(before_id) = before {
        qb.push(" WHERE episode_id < ");
        qb.push_bind(before_id);
    } else if let Some(after_id) = after {
        qb.push(" WHERE episode_id > ");
        qb.push_bind(after_id);
    }

    qb.push(format!(" ORDER BY episode_id {order} LIMIT "));
    qb.push_bind(limit as i64);

    // CTE 2: Aggregate only the candidate episodes
    qb.push(format!(
        r")
            ) t
        ),
        all_inferences AS (
            SELECT episode_id, id, created_at FROM tensorzero.chat_inferences
            WHERE episode_id IN (SELECT episode_id FROM candidate_episode_ids)
            UNION ALL
            SELECT episode_id, id, created_at FROM tensorzero.json_inferences
            WHERE episode_id IN (SELECT episode_id FROM candidate_episode_ids)
        )
        SELECT
            episode_id,
            COUNT(*)::BIGINT as count,
            MIN(created_at) as start_time,
            MAX(created_at) as end_time,
            tensorzero.max_uuid(id) as last_inference_id
        FROM all_inferences
        GROUP BY episode_id
        ORDER BY episode_id {order} LIMIT "
    ));
    qb.push_bind(limit as i64);

    EpisodeTableQueryResult {
        query_builder: qb,
        reverse_results,
    }
}

/// Builds an episode listing query with filters (function_name and/or inference filters).
///
/// Pushes filters, pagination, and limit into each per-table subquery so that
/// the database can use indexes efficiently without scanning unneeded rows.
/// Each inner query returns at most `limit` episode_ids; the outer DISTINCT +
/// LIMIT deduplicates across tables and picks the final top `limit`.
fn build_filtered_episode_query(
    config: &Config,
    limit: u32,
    before: Option<Uuid>,
    after: Option<Uuid>,
    function_name: Option<&str>,
    filters: Option<&InferenceFilter>,
) -> Result<EpisodeTableQueryResult, Error> {
    let reverse_results = after.is_some();
    let order = if after.is_some() { "ASC" } else { "DESC" };

    let mut qb = QueryBuilder::<sqlx::Postgres>::new("");

    // CTE 1: Find top episode_ids matching filters with pagination and limit.
    qb.push(
        r"
        WITH top_episodes AS (
            SELECT DISTINCT episode_id FROM (
                (SELECT DISTINCT i.episode_id FROM tensorzero.chat_inferences i
                WHERE 1=1",
    );
    if let Some(fn_name) = function_name {
        qb.push(" AND i.function_name = ");
        qb.push_bind(fn_name.to_string());
    }
    apply_inference_filter(&mut qb, filters, config)?;
    if let Some(before_id) = before {
        qb.push(" AND i.episode_id < ");
        qb.push_bind(before_id);
    } else if let Some(after_id) = after {
        qb.push(" AND i.episode_id > ");
        qb.push_bind(after_id);
    }
    qb.push(format!(" ORDER BY i.episode_id {order} LIMIT "));
    qb.push_bind(limit as i64);

    qb.push(
        r")
                UNION ALL
                (SELECT DISTINCT i.episode_id FROM tensorzero.json_inferences i
                WHERE 1=1",
    );
    if let Some(fn_name) = function_name {
        qb.push(" AND i.function_name = ");
        qb.push_bind(fn_name.to_string());
    }
    apply_inference_filter(&mut qb, filters, config)?;
    if let Some(before_id) = before {
        qb.push(" AND i.episode_id < ");
        qb.push_bind(before_id);
    } else if let Some(after_id) = after {
        qb.push(" AND i.episode_id > ");
        qb.push_bind(after_id);
    }
    qb.push(format!(" ORDER BY i.episode_id {order} LIMIT "));
    qb.push_bind(limit as i64);

    qb.push(format!(
        r")
            ) t
            ORDER BY episode_id {order} LIMIT "
    ));
    qb.push_bind(limit as i64);

    // CTE 2: Get all inferences for the top episodes (for full aggregation)
    qb.push(format!(
        r"
        ),
        all_inferences AS (
            SELECT episode_id, id, created_at FROM tensorzero.chat_inferences
            WHERE episode_id IN (SELECT episode_id FROM top_episodes)
            UNION ALL
            SELECT episode_id, id, created_at FROM tensorzero.json_inferences
            WHERE episode_id IN (SELECT episode_id FROM top_episodes)
        )
        SELECT
            episode_id,
            COUNT(*)::BIGINT as count,
            MIN(created_at) as start_time,
            MAX(created_at) as end_time,
            tensorzero.max_uuid(id) as last_inference_id
        FROM all_inferences
        GROUP BY episode_id
        ORDER BY episode_id {order}
        "
    ));

    Ok(EpisodeTableQueryResult {
        query_builder: qb,
        reverse_results,
    })
}

// =====================================================================
// Query execution functions
// =====================================================================

async fn query_episode_table_impl(
    pool: &PgPool,
    config: &Config,
    limit: u32,
    before: Option<Uuid>,
    after: Option<Uuid>,
    function_name: Option<&str>,
    filters: Option<&InferenceFilter>,
) -> Result<Vec<EpisodeByIdRow>, Error> {
    let EpisodeTableQueryResult {
        mut query_builder,
        reverse_results,
    } = build_query_episode_table(config, limit, before, after, function_name, filters)?;

    let mut results: Vec<EpisodeByIdRow> = query_builder
        .build_query_as::<EpisodeByIdRow>()
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to query episode table: {e}"),
            })
        })?;

    // For "after" pagination, we queried in ASC order but want to return DESC
    // Also, we want the final order to be DESC (most recent first)
    if reverse_results {
        results.reverse();
    }

    Ok(results)
}

async fn query_episode_table_bounds_impl(pool: &PgPool) -> Result<TableBoundsWithCount, Error> {
    // Bounds: find the min/max episode_id across chat and json inferences.
    // Count: pg_class.reltuples for an approximate total (updated by autovacuum/analyze).
    //
    // TODO(#6472): Implement accurate episode count.
    sqlx::query_as(
        r"
        SELECT
            (SELECT episode_id FROM (
                (SELECT episode_id FROM tensorzero.chat_inferences ORDER BY episode_id ASC LIMIT 1)
                UNION ALL
                (SELECT episode_id FROM tensorzero.json_inferences ORDER BY episode_id ASC LIMIT 1)
            ) t ORDER BY episode_id ASC LIMIT 1) as first_id,
            (SELECT episode_id FROM (
                (SELECT episode_id FROM tensorzero.chat_inferences ORDER BY episode_id DESC LIMIT 1)
                UNION ALL
                (SELECT episode_id FROM tensorzero.json_inferences ORDER BY episode_id DESC LIMIT 1)
            ) t ORDER BY episode_id DESC LIMIT 1) as last_id,
            COALESCE((
                SELECT SUM(GREATEST(c.reltuples, 0))::BIGINT
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                JOIN pg_inherits i ON i.inhrelid = c.oid
                JOIN pg_class parent ON parent.oid = i.inhparent
                JOIN pg_namespace pn ON pn.oid = parent.relnamespace
                WHERE pn.nspname = 'tensorzero'
                AND parent.relname IN ('chat_inferences', 'json_inferences')
            ), 0) as count
        ",
    )
    .fetch_one(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to query episode table bounds: {e}"),
        })
    })
}

// =====================================================================
// Unit tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test_helpers::assert_query_equals;

    #[test]
    fn test_build_query_episode_table_no_pagination() {
        let config = Config::default();
        let result = build_query_episode_table(&config, 10, None, None, None, None).unwrap();
        assert!(
            !result.reverse_results,
            "Should not reverse results without pagination"
        );

        assert_query_equals(
            result.query_builder.sql().as_str(),
            r"
            WITH candidate_episode_ids AS (
                SELECT DISTINCT episode_id FROM (
                    (SELECT DISTINCT episode_id FROM tensorzero.chat_inferences ORDER BY episode_id DESC LIMIT $1)
                    UNION ALL
                    (SELECT DISTINCT episode_id FROM tensorzero.json_inferences ORDER BY episode_id DESC LIMIT $2)
                ) t
            ),
            all_inferences AS (
                SELECT episode_id, id, created_at FROM tensorzero.chat_inferences
                WHERE episode_id IN (SELECT episode_id FROM candidate_episode_ids)
                UNION ALL
                SELECT episode_id, id, created_at FROM tensorzero.json_inferences
                WHERE episode_id IN (SELECT episode_id FROM candidate_episode_ids)
            )
            SELECT
                episode_id,
                COUNT(*)::BIGINT as count,
                MIN(created_at) as start_time,
                MAX(created_at) as end_time,
                tensorzero.max_uuid(id) as last_inference_id
            FROM all_inferences
            GROUP BY episode_id
            ORDER BY episode_id DESC LIMIT $3
            ",
        );
    }

    #[test]
    fn test_build_query_episode_table_with_before() {
        let config = Config::default();
        let before_id = Uuid::now_v7();
        let result =
            build_query_episode_table(&config, 20, Some(before_id), None, None, None).unwrap();
        assert!(
            !result.reverse_results,
            "Should not reverse results with before pagination"
        );

        assert_query_equals(
            result.query_builder.sql().as_str(),
            r"
            WITH candidate_episode_ids AS (
                SELECT DISTINCT episode_id FROM (
                    (SELECT DISTINCT episode_id FROM tensorzero.chat_inferences WHERE episode_id < $1 ORDER BY episode_id DESC LIMIT $2)
                    UNION ALL
                    (SELECT DISTINCT episode_id FROM tensorzero.json_inferences WHERE episode_id < $3 ORDER BY episode_id DESC LIMIT $4)
                ) t
            ),
            all_inferences AS (
                SELECT episode_id, id, created_at FROM tensorzero.chat_inferences
                WHERE episode_id IN (SELECT episode_id FROM candidate_episode_ids)
                UNION ALL
                SELECT episode_id, id, created_at FROM tensorzero.json_inferences
                WHERE episode_id IN (SELECT episode_id FROM candidate_episode_ids)
            )
            SELECT
                episode_id,
                COUNT(*)::BIGINT as count,
                MIN(created_at) as start_time,
                MAX(created_at) as end_time,
                tensorzero.max_uuid(id) as last_inference_id
            FROM all_inferences
            GROUP BY episode_id
            ORDER BY episode_id DESC LIMIT $5
            ",
        );
    }

    #[test]
    fn test_build_query_episode_table_with_after() {
        let config = Config::default();
        let after_id = Uuid::now_v7();
        let result =
            build_query_episode_table(&config, 15, None, Some(after_id), None, None).unwrap();
        assert!(
            result.reverse_results,
            "Should reverse results with after pagination"
        );

        assert_query_equals(
            result.query_builder.sql().as_str(),
            r"
            WITH candidate_episode_ids AS (
                SELECT DISTINCT episode_id FROM (
                    (SELECT DISTINCT episode_id FROM tensorzero.chat_inferences WHERE episode_id > $1 ORDER BY episode_id ASC LIMIT $2)
                    UNION ALL
                    (SELECT DISTINCT episode_id FROM tensorzero.json_inferences WHERE episode_id > $3 ORDER BY episode_id ASC LIMIT $4)
                ) t
            ),
            all_inferences AS (
                SELECT episode_id, id, created_at FROM tensorzero.chat_inferences
                WHERE episode_id IN (SELECT episode_id FROM candidate_episode_ids)
                UNION ALL
                SELECT episode_id, id, created_at FROM tensorzero.json_inferences
                WHERE episode_id IN (SELECT episode_id FROM candidate_episode_ids)
            )
            SELECT
                episode_id,
                COUNT(*)::BIGINT as count,
                MIN(created_at) as start_time,
                MAX(created_at) as end_time,
                tensorzero.max_uuid(id) as last_inference_id
            FROM all_inferences
            GROUP BY episode_id
            ORDER BY episode_id ASC LIMIT $5
            ",
        );
    }

    #[test]
    fn test_build_query_episode_table_rejects_both_before_and_after() {
        let config = Config::default();
        let before_id = Uuid::now_v7();
        let after_id = Uuid::now_v7();
        let result =
            build_query_episode_table(&config, 10, Some(before_id), Some(after_id), None, None);

        match result {
            Ok(_) => panic!("Should error when both before and after are specified"),
            Err(err) => {
                assert!(
                    err.to_string()
                        .contains("Cannot specify both before and after"),
                    "Error message should mention that both before and after cannot be specified"
                );
            }
        }
    }

    #[test]
    fn test_build_query_episode_table_with_function_name() {
        let config = Config::default();
        let result =
            build_query_episode_table(&config, 10, None, None, Some("my_function"), None).unwrap();
        assert!(
            !result.reverse_results,
            "Should not reverse results without pagination"
        );

        assert_query_equals(
            result.query_builder.sql().as_str(),
            r"
            WITH top_episodes AS (
                SELECT DISTINCT episode_id FROM (
                    (SELECT DISTINCT i.episode_id FROM tensorzero.chat_inferences i
                    WHERE 1=1 AND i.function_name = $1 ORDER BY i.episode_id DESC LIMIT $2)
                    UNION ALL
                    (SELECT DISTINCT i.episode_id FROM tensorzero.json_inferences i
                    WHERE 1=1 AND i.function_name = $3 ORDER BY i.episode_id DESC LIMIT $4)
                ) t
                ORDER BY episode_id DESC LIMIT $5
            ),
            all_inferences AS (
                SELECT episode_id, id, created_at FROM tensorzero.chat_inferences
                WHERE episode_id IN (SELECT episode_id FROM top_episodes)
                UNION ALL
                SELECT episode_id, id, created_at FROM tensorzero.json_inferences
                WHERE episode_id IN (SELECT episode_id FROM top_episodes)
            )
            SELECT
                episode_id,
                COUNT(*)::BIGINT as count,
                MIN(created_at) as start_time,
                MAX(created_at) as end_time,
                tensorzero.max_uuid(id) as last_inference_id
            FROM all_inferences
            GROUP BY episode_id
            ORDER BY episode_id DESC
            ",
        );
    }

    #[test]
    fn test_build_query_episode_table_with_function_name_and_pagination() {
        let config = Config::default();
        let before_id = Uuid::now_v7();
        let result = build_query_episode_table(
            &config,
            10,
            Some(before_id),
            None,
            Some("my_function"),
            None,
        )
        .unwrap();

        assert_query_equals(
            result.query_builder.sql().as_str(),
            r"
            WITH top_episodes AS (
                SELECT DISTINCT episode_id FROM (
                    (SELECT DISTINCT i.episode_id FROM tensorzero.chat_inferences i
                    WHERE 1=1 AND i.function_name = $1 AND i.episode_id < $2 ORDER BY i.episode_id DESC LIMIT $3)
                    UNION ALL
                    (SELECT DISTINCT i.episode_id FROM tensorzero.json_inferences i
                    WHERE 1=1 AND i.function_name = $4 AND i.episode_id < $5 ORDER BY i.episode_id DESC LIMIT $6)
                ) t
                ORDER BY episode_id DESC LIMIT $7
            ),
            all_inferences AS (
                SELECT episode_id, id, created_at FROM tensorzero.chat_inferences
                WHERE episode_id IN (SELECT episode_id FROM top_episodes)
                UNION ALL
                SELECT episode_id, id, created_at FROM tensorzero.json_inferences
                WHERE episode_id IN (SELECT episode_id FROM top_episodes)
            )
            SELECT
                episode_id,
                COUNT(*)::BIGINT as count,
                MIN(created_at) as start_time,
                MAX(created_at) as end_time,
                tensorzero.max_uuid(id) as last_inference_id
            FROM all_inferences
            GROUP BY episode_id
            ORDER BY episode_id DESC
            ",
        );
    }

    #[test]
    fn test_build_query_episode_table_with_tag_filter() {
        use crate::db::clickhouse::query_builder::TagFilter;
        use crate::endpoints::stored_inferences::v1::types::TagComparisonOperator;

        let config = Config::default();
        let filter = InferenceFilter::Tag(TagFilter {
            key: "env".to_string(),
            value: "prod".to_string(),
            comparison_operator: TagComparisonOperator::Equal,
        });
        let result =
            build_query_episode_table(&config, 10, None, None, None, Some(&filter)).unwrap();

        assert_query_equals(
            result.query_builder.sql().as_str(),
            r"
            WITH top_episodes AS (
                SELECT DISTINCT episode_id FROM (
                    (SELECT DISTINCT i.episode_id FROM tensorzero.chat_inferences i
                    WHERE 1=1 AND i.tags @> jsonb_build_object($1, $2) ORDER BY i.episode_id DESC LIMIT $3)
                    UNION ALL
                    (SELECT DISTINCT i.episode_id FROM tensorzero.json_inferences i
                    WHERE 1=1 AND i.tags @> jsonb_build_object($4, $5) ORDER BY i.episode_id DESC LIMIT $6)
                ) t
                ORDER BY episode_id DESC LIMIT $7
            ),
            all_inferences AS (
                SELECT episode_id, id, created_at FROM tensorzero.chat_inferences
                WHERE episode_id IN (SELECT episode_id FROM top_episodes)
                UNION ALL
                SELECT episode_id, id, created_at FROM tensorzero.json_inferences
                WHERE episode_id IN (SELECT episode_id FROM top_episodes)
            )
            SELECT
                episode_id,
                COUNT(*)::BIGINT as count,
                MIN(created_at) as start_time,
                MAX(created_at) as end_time,
                tensorzero.max_uuid(id) as last_inference_id
            FROM all_inferences
            GROUP BY episode_id
            ORDER BY episode_id DESC
            ",
        );
    }

    #[test]
    fn test_build_query_episode_table_with_function_name_and_filter() {
        use crate::db::clickhouse::query_builder::TagFilter;
        use crate::endpoints::stored_inferences::v1::types::TagComparisonOperator;

        let config = Config::default();
        let filter = InferenceFilter::Tag(TagFilter {
            key: "env".to_string(),
            value: "prod".to_string(),
            comparison_operator: TagComparisonOperator::Equal,
        });
        let result =
            build_query_episode_table(&config, 10, None, None, Some("my_function"), Some(&filter))
                .unwrap();

        assert_query_equals(
            result.query_builder.sql().as_str(),
            r"
            WITH top_episodes AS (
                SELECT DISTINCT episode_id FROM (
                    (SELECT DISTINCT i.episode_id FROM tensorzero.chat_inferences i
                    WHERE 1=1 AND i.function_name = $1 AND i.tags @> jsonb_build_object($2, $3) ORDER BY i.episode_id DESC LIMIT $4)
                    UNION ALL
                    (SELECT DISTINCT i.episode_id FROM tensorzero.json_inferences i
                    WHERE 1=1 AND i.function_name = $5 AND i.tags @> jsonb_build_object($6, $7) ORDER BY i.episode_id DESC LIMIT $8)
                ) t
                ORDER BY episode_id DESC LIMIT $9
            ),
            all_inferences AS (
                SELECT episode_id, id, created_at FROM tensorzero.chat_inferences
                WHERE episode_id IN (SELECT episode_id FROM top_episodes)
                UNION ALL
                SELECT episode_id, id, created_at FROM tensorzero.json_inferences
                WHERE episode_id IN (SELECT episode_id FROM top_episodes)
            )
            SELECT
                episode_id,
                COUNT(*)::BIGINT as count,
                MIN(created_at) as start_time,
                MAX(created_at) as end_time,
                tensorzero.max_uuid(id) as last_inference_id
            FROM all_inferences
            GROUP BY episode_id
            ORDER BY episode_id DESC
            ",
        );
    }
}
