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
    let has_filters = function_name.is_some() || filters.is_some();

    // For pagination, we need to handle the before/after cases differently
    let (where_clause, order_direction, reverse_results) = match (before, after) {
        (Some(before_id), None) => {
            // Get episodes with IDs less than the cursor
            (Some(("episode_id < ", before_id)), "DESC", false)
        }
        (None, Some(after_id)) => {
            // Get episodes with IDs greater than the cursor
            // Query in ASC order, then reverse the results
            (Some(("episode_id > ", after_id)), "ASC", true)
        }
        (None, None) => (None, "DESC", false),
        // Both before and after specified
        (Some(_), Some(_)) => {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Cannot specify both before and after in query_episode_table".to_string(),
            }));
        }
    };

    // Build the query using a CTE to combine both inference tables
    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("");

    if has_filters {
        // When filtering, alias each table as `i` so that apply_inference_filter
        // can reference `i.*` columns (i.tags, i.created_at, i.id, i.episode_id).
        query_builder.push(
            r"
        WITH all_inferences AS (
            SELECT i.episode_id, i.id, i.created_at FROM tensorzero.chat_inferences i
            WHERE 1=1",
        );

        if let Some(fn_name) = function_name {
            query_builder.push(" AND i.function_name = ");
            query_builder.push_bind(fn_name.to_string());
        }

        apply_inference_filter(&mut query_builder, filters, config)?;

        query_builder.push(
            r"
            UNION ALL
            SELECT i.episode_id, i.id, i.created_at FROM tensorzero.json_inferences i
            WHERE 1=1",
        );

        if let Some(fn_name) = function_name {
            query_builder.push(" AND i.function_name = ");
            query_builder.push_bind(fn_name.to_string());
        }

        apply_inference_filter(&mut query_builder, filters, config)?;

        query_builder.push(
            r"
        ),",
        );
    } else {
        query_builder.push(
            r"
        WITH all_inferences AS (
            SELECT episode_id, id, created_at FROM tensorzero.chat_inferences
            UNION ALL
            SELECT episode_id, id, created_at FROM tensorzero.json_inferences
        ),",
        );
    }

    query_builder.push(
        r"
        episode_aggregates AS (
            SELECT
                episode_id,
                COUNT(*)::BIGINT as count,
                MIN(created_at) as start_time,
                MAX(created_at) as end_time,
                tensorzero.max_uuid(id) as last_inference_id
            FROM all_inferences
            GROUP BY episode_id
        )
        SELECT
            episode_id,
            count,
            start_time,
            end_time,
            last_inference_id
        FROM episode_aggregates
        ",
    );

    // Add WHERE clause if we have pagination
    if let Some((comparison, cursor_id)) = where_clause {
        query_builder.push("WHERE ");
        query_builder.push(comparison);
        query_builder.push_bind(cursor_id);
    }

    // Add ORDER BY
    query_builder.push(" ORDER BY episode_id ");
    query_builder.push(order_direction);

    // Add LIMIT
    query_builder.push(" LIMIT ");
    query_builder.push_bind(limit as i64);

    Ok(EpisodeTableQueryResult {
        query_builder,
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
    // Use a CTE to combine both inference tables and get bounds
    // Use subqueries with ORDER BY/LIMIT instead of MIN/MAX since Postgres doesn't support MIN/MAX on UUID
    sqlx::query_as(
        r"
        WITH all_episodes AS (
            SELECT DISTINCT episode_id FROM tensorzero.chat_inferences
            UNION
            SELECT DISTINCT episode_id FROM tensorzero.json_inferences
        )
        SELECT
            (SELECT episode_id FROM all_episodes ORDER BY episode_id ASC LIMIT 1) as first_id,
            (SELECT episode_id FROM all_episodes ORDER BY episode_id DESC LIMIT 1) as last_id,
            COUNT(*)::BIGINT as count
        FROM all_episodes
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
            WITH all_inferences AS (
                SELECT episode_id, id, created_at FROM tensorzero.chat_inferences
                UNION ALL
                SELECT episode_id, id, created_at FROM tensorzero.json_inferences
            ),
            episode_aggregates AS (
                SELECT
                    episode_id,
                    COUNT(*)::BIGINT as count,
                    MIN(created_at) as start_time,
                    MAX(created_at) as end_time,
                    tensorzero.max_uuid(id) as last_inference_id
                FROM all_inferences
                GROUP BY episode_id
            )
            SELECT
                episode_id,
                count,
                start_time,
                end_time,
                last_inference_id
            FROM episode_aggregates
            ORDER BY episode_id DESC
            LIMIT $1
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
            WITH all_inferences AS (
                SELECT episode_id, id, created_at FROM tensorzero.chat_inferences
                UNION ALL
                SELECT episode_id, id, created_at FROM tensorzero.json_inferences
            ),
            episode_aggregates AS (
                SELECT
                    episode_id,
                    COUNT(*)::BIGINT as count,
                    MIN(created_at) as start_time,
                    MAX(created_at) as end_time,
                    tensorzero.max_uuid(id) as last_inference_id
                FROM all_inferences
                GROUP BY episode_id
            )
            SELECT
                episode_id,
                count,
                start_time,
                end_time,
                last_inference_id
            FROM episode_aggregates
            WHERE episode_id < $1
            ORDER BY episode_id DESC
            LIMIT $2
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
            WITH all_inferences AS (
                SELECT episode_id, id, created_at FROM tensorzero.chat_inferences
                UNION ALL
                SELECT episode_id, id, created_at FROM tensorzero.json_inferences
            ),
            episode_aggregates AS (
                SELECT
                    episode_id,
                    COUNT(*)::BIGINT as count,
                    MIN(created_at) as start_time,
                    MAX(created_at) as end_time,
                    tensorzero.max_uuid(id) as last_inference_id
                FROM all_inferences
                GROUP BY episode_id
            )
            SELECT
                episode_id,
                count,
                start_time,
                end_time,
                last_inference_id
            FROM episode_aggregates
            WHERE episode_id > $1
            ORDER BY episode_id ASC
            LIMIT $2
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
            WITH all_inferences AS (
                SELECT i.episode_id, i.id, i.created_at FROM tensorzero.chat_inferences i
                WHERE 1=1 AND i.function_name = $1
                UNION ALL
                SELECT i.episode_id, i.id, i.created_at FROM tensorzero.json_inferences i
                WHERE 1=1 AND i.function_name = $2
            ),
            episode_aggregates AS (
                SELECT
                    episode_id,
                    COUNT(*)::BIGINT as count,
                    MIN(created_at) as start_time,
                    MAX(created_at) as end_time,
                    tensorzero.max_uuid(id) as last_inference_id
                FROM all_inferences
                GROUP BY episode_id
            )
            SELECT
                episode_id,
                count,
                start_time,
                end_time,
                last_inference_id
            FROM episode_aggregates
            ORDER BY episode_id DESC
            LIMIT $3
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
            WITH all_inferences AS (
                SELECT i.episode_id, i.id, i.created_at FROM tensorzero.chat_inferences i
                WHERE 1=1 AND i.function_name = $1
                UNION ALL
                SELECT i.episode_id, i.id, i.created_at FROM tensorzero.json_inferences i
                WHERE 1=1 AND i.function_name = $2
            ),
            episode_aggregates AS (
                SELECT
                    episode_id,
                    COUNT(*)::BIGINT as count,
                    MIN(created_at) as start_time,
                    MAX(created_at) as end_time,
                    tensorzero.max_uuid(id) as last_inference_id
                FROM all_inferences
                GROUP BY episode_id
            )
            SELECT
                episode_id,
                count,
                start_time,
                end_time,
                last_inference_id
            FROM episode_aggregates
            WHERE episode_id < $3
            ORDER BY episode_id DESC
            LIMIT $4
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
            WITH all_inferences AS (
                SELECT i.episode_id, i.id, i.created_at FROM tensorzero.chat_inferences i
                WHERE 1=1 AND (i.tags ? $1 AND i.tags->>$2 = $3)
                UNION ALL
                SELECT i.episode_id, i.id, i.created_at FROM tensorzero.json_inferences i
                WHERE 1=1 AND (i.tags ? $4 AND i.tags->>$5 = $6)
            ),
            episode_aggregates AS (
                SELECT
                    episode_id,
                    COUNT(*)::BIGINT as count,
                    MIN(created_at) as start_time,
                    MAX(created_at) as end_time,
                    tensorzero.max_uuid(id) as last_inference_id
                FROM all_inferences
                GROUP BY episode_id
            )
            SELECT
                episode_id,
                count,
                start_time,
                end_time,
                last_inference_id
            FROM episode_aggregates
            ORDER BY episode_id DESC
            LIMIT $7
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
            WITH all_inferences AS (
                SELECT i.episode_id, i.id, i.created_at FROM tensorzero.chat_inferences i
                WHERE 1=1 AND i.function_name = $1 AND (i.tags ? $2 AND i.tags->>$3 = $4)
                UNION ALL
                SELECT i.episode_id, i.id, i.created_at FROM tensorzero.json_inferences i
                WHERE 1=1 AND i.function_name = $5 AND (i.tags ? $6 AND i.tags->>$7 = $8)
            ),
            episode_aggregates AS (
                SELECT
                    episode_id,
                    COUNT(*)::BIGINT as count,
                    MIN(created_at) as start_time,
                    MAX(created_at) as end_time,
                    tensorzero.max_uuid(id) as last_inference_id
                FROM all_inferences
                GROUP BY episode_id
            )
            SELECT
                episode_id,
                count,
                start_time,
                end_time,
                last_inference_id
            FROM episode_aggregates
            ORDER BY episode_id DESC
            LIMIT $9
            ",
        );
    }
}
