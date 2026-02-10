//! EpisodeQueries implementation for Postgres.
//!
//! This module implements episode queries for the Postgres database.

use async_trait::async_trait;
use sqlx::{PgPool, QueryBuilder};
use uuid::Uuid;

use crate::db::{EpisodeByIdRow, EpisodeQueries, TableBoundsWithCount};
use crate::error::{Error, ErrorDetails};

use super::PostgresConnectionInfo;

#[async_trait]
impl EpisodeQueries for PostgresConnectionInfo {
    async fn query_episode_table(
        &self,
        limit: u32,
        before: Option<Uuid>,
        after: Option<Uuid>,
    ) -> Result<Vec<EpisodeByIdRow>, Error> {
        let pool = self.get_pool_result()?;
        query_episode_table_impl(pool, limit, before, after).await
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

/// Builds a query to fetch episodes with pagination.
///
/// Returns an error if both `before` and `after` are specified.
fn build_query_episode_table(
    limit: u32,
    before: Option<Uuid>,
    after: Option<Uuid>,
) -> Result<EpisodeTableQueryResult, Error> {
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
        // Both before and after specified - already handled above, but Rust requires exhaustive matching
        (Some(_), Some(_)) => {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Cannot specify both before and after in query_episode_table".to_string(),
            }));
        }
    };

    // Build the query using a CTE to combine both inference tables
    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("");
    query_builder.push(
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
    limit: u32,
    before: Option<Uuid>,
    after: Option<Uuid>,
) -> Result<Vec<EpisodeByIdRow>, Error> {
    let EpisodeTableQueryResult {
        mut query_builder,
        reverse_results,
    } = build_query_episode_table(limit, before, after)?;

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
        let result = build_query_episode_table(10, None, None).unwrap();
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
        let before_id = Uuid::now_v7();
        let result = build_query_episode_table(20, Some(before_id), None).unwrap();
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
        let after_id = Uuid::now_v7();
        let result = build_query_episode_table(15, None, Some(after_id)).unwrap();
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
        let before_id = Uuid::now_v7();
        let after_id = Uuid::now_v7();
        let result = build_query_episode_table(10, Some(before_id), Some(after_id));

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
}
