//! Dataset queries for Postgres.
//!
//! This module implements both read and write operations for datapoint tables in Postgres.

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use chrono::{DateTime, NaiveDateTime, Utc};
use sqlx::postgres::PgRow;
use sqlx::types::Json;
use sqlx::{PgPool, QueryBuilder, Row};
use uuid::Uuid;

use crate::config::snapshot::SnapshotHash;
use crate::db::clickhouse::query_builder::{
    DatapointFilter, TagComparisonOperator, TagFilter, TimeComparisonOperator, TimeFilter,
};
use crate::db::datasets::{
    DEFAULT_ALLOW_STALE_IN_GET_DATAPOINT, DatasetMetadata, DatasetQueries, GetDatapointParams,
    GetDatapointsParams, GetDatasetMetadataParams,
};
use crate::db::query_helpers::{json_double_escape_string_without_quotes, uuid_to_datetime};
use crate::db::stored_datapoint::{
    StoredChatInferenceDatapoint, StoredDatapoint, StoredJsonInferenceDatapoint,
};
use crate::endpoints::datasets::v1::types::{DatapointOrderBy, DatapointOrderByTerm};
use crate::endpoints::datasets::{CLICKHOUSE_DATETIME_FORMAT, validate_dataset_name};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::stored_input::StoredInput;
use crate::inference::types::{ContentBlockChatOutput, JsonInferenceOutput};
use crate::tool::ToolCallConfigDatabaseInsert;
use crate::tool::config::AllowedTools;
use crate::tool::types::{ProviderTool, Tool};
use crate::tool::wire::ToolChoice;

use super::PostgresConnectionInfo;

/// Parses a datetime string in ClickHouse format to DateTime<Utc>.
/// Returns None if the input is None, or an error if parsing fails.
fn parse_clickhouse_datetime(s: &str) -> Result<DateTime<Utc>, Error> {
    NaiveDateTime::parse_from_str(s, CLICKHOUSE_DATETIME_FORMAT)
        .map(|naive| naive.and_utc())
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to parse datetime string '{s}': {e}"),
            })
        })
}

// =====================================================================
// DatasetQueries trait implementation
// =====================================================================

#[async_trait]
impl DatasetQueries for PostgresConnectionInfo {
    async fn get_dataset_metadata(
        &self,
        params: &GetDatasetMetadataParams,
    ) -> Result<Vec<DatasetMetadata>, Error> {
        let pool = self.get_pool_result()?;
        let mut qb = build_get_dataset_metadata_query(params);

        let rows = qb.build().fetch_all(pool).await?;

        Ok(rows
            .into_iter()
            .map(|row| {
                let last_updated: DateTime<Utc> = row.get("last_updated");
                DatasetMetadata {
                    dataset_name: row.get("dataset_name"),
                    count: row.get::<i32, _>("count") as u32,
                    last_updated: last_updated.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
                }
            })
            .collect())
    }

    async fn insert_datapoints(&self, datapoints: &[StoredDatapoint]) -> Result<u64, Error> {
        let pool = self.get_pool_result()?;

        // Separate chat and JSON datapoints
        let mut chat_datapoints: Vec<&StoredChatInferenceDatapoint> = Vec::new();
        let mut json_datapoints: Vec<&StoredJsonInferenceDatapoint> = Vec::new();

        for datapoint in datapoints {
            match datapoint {
                StoredDatapoint::Chat(chat) => chat_datapoints.push(chat),
                StoredDatapoint::Json(json) => json_datapoints.push(json),
            }
        }

        let (chat_count, json_count) = tokio::join!(
            insert_chat_datapoints(pool, &chat_datapoints),
            insert_json_datapoints(pool, &json_datapoints),
        );

        Ok(chat_count? + json_count?)
    }

    async fn count_datapoints_for_dataset(
        &self,
        dataset_name: &str,
        function_name: Option<&str>,
    ) -> Result<u64, Error> {
        let pool = self.get_pool_result()?;
        let mut qb = build_count_datapoints_query(dataset_name, function_name);

        let row = qb.build().fetch_one(pool).await?;
        let total: i64 = row.get("total");

        Ok(total as u64)
    }

    async fn get_datapoint(&self, params: &GetDatapointParams) -> Result<StoredDatapoint, Error> {
        let allow_stale = params
            .allow_stale
            .unwrap_or(DEFAULT_ALLOW_STALE_IN_GET_DATAPOINT);

        let mut datapoints = self
            .get_datapoints(&GetDatapointsParams {
                dataset_name: Some(params.dataset_name.clone()),
                function_name: None,
                ids: Some(vec![params.datapoint_id]),
                limit: 1,
                offset: 0,
                allow_stale,
                filter: None,
                order_by: None,
                search_query_experimental: None,
            })
            .await?;

        if datapoints.is_empty() {
            return Err(Error::new(ErrorDetails::DatapointNotFound {
                dataset_name: params.dataset_name.clone(),
                datapoint_id: params.datapoint_id,
            }));
        }

        let first_datapoint = datapoints.swap_remove(0);
        Ok(first_datapoint)
    }

    async fn get_datapoints(
        &self,
        params: &GetDatapointsParams,
    ) -> Result<Vec<StoredDatapoint>, Error> {
        let pool = self.get_pool_result()?;

        // If neither IDs nor dataset are provided, reject the query.
        if params.dataset_name.is_none() && params.ids.is_none() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "At least one of dataset_name or ids must be provided".to_string(),
            }));
        }

        // If IDs are provided, they must not be empty.
        if let Some(ids_vec) = &params.ids
            && ids_vec.is_empty()
        {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "ids must not be an empty list".to_string(),
            }));
        }

        // Validate that if SearchRelevance is used, search_query_experimental must be present
        if let Some(order_by_vec) = &params.order_by {
            for order_spec in order_by_vec {
                if matches!(order_spec.term, DatapointOrderByTerm::SearchRelevance)
                    && params.search_query_experimental.is_none()
                {
                    return Err(Error::new(ErrorDetails::InvalidRequest {
                        message:
                            "OrderBy::SearchRelevance requires search_query_experimental to be provided"
                                .to_string(),
                    }));
                }
            }
        }

        // Use UNION ALL query to push sorting and pagination to the database
        let mut qb = build_datapoints_union_query(params)?;
        let results: Vec<StoredDatapoint> = qb.build_query_as().fetch_all(pool).await?;

        Ok(results)
    }

    async fn delete_datapoints(
        &self,
        dataset_name: &str,
        datapoint_ids: Option<&[Uuid]>,
    ) -> Result<u64, Error> {
        let pool = self.get_pool_result()?;

        // Validate input
        if let Some(ids) = datapoint_ids
            && ids.is_empty()
        {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "If datapoint_ids are provided as a vector, it must be non-empty"
                    .to_string(),
            }));
        }

        // Delete (soft) from both tables in parallel
        let (chat_deleted, json_deleted) = tokio::join!(
            delete_chat_datapoints(pool, dataset_name, datapoint_ids),
            delete_json_datapoints(pool, dataset_name, datapoint_ids),
        );

        Ok(chat_deleted? + json_deleted?)
    }

    async fn clone_datapoints(
        &self,
        target_dataset_name: &str,
        source_datapoint_ids: &[Uuid],
        id_mappings: &HashMap<Uuid, Uuid>,
    ) -> Result<Vec<Option<Uuid>>, Error> {
        let pool = self.get_pool_result()?;

        if source_datapoint_ids.is_empty() {
            return Ok(vec![]);
        }

        // Convert to ordered Vec using source_datapoint_ids for consistent ordering
        let mappings: Vec<(Uuid, Uuid)> = source_datapoint_ids
            .iter()
            .filter_map(|src| id_mappings.get(src).map(|dst| (*src, *dst)))
            .collect();

        // Clone from both tables in parallel
        let (chat_result, json_result) = tokio::join!(
            clone_chat_datapoints(pool, target_dataset_name, &mappings),
            clone_json_datapoints(pool, target_dataset_name, &mappings),
        );

        chat_result?;
        json_result?;

        // Verify which new_ids were actually created
        let new_ids: Vec<Uuid> = mappings.iter().map(|(_, new)| *new).collect();
        let created_ids = verify_created_ids(pool, &new_ids).await?;

        // Map results based on which new_ids were created
        let results: Vec<Option<Uuid>> = mappings
            .iter()
            .map(|(source_id, new_id)| {
                if created_ids.contains(new_id) {
                    Some(*new_id)
                } else {
                    tracing::warn!(
                        "Failed to clone datapoint (likely does not exist): {source_id}"
                    );
                    None
                }
            })
            .collect();

        Ok(results)
    }
}

// =====================================================================
// Query builder functions (for unit testing)
// =====================================================================

/// Builds a query to get dataset metadata.
fn build_get_dataset_metadata_query(
    params: &GetDatasetMetadataParams,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        WITH unioned_datasets AS (
            SELECT
                dataset_name,
                COUNT(*) AS count,
                MAX(updated_at) AS last_updated
            FROM tensorzero.chat_datapoints
            WHERE staled_at IS NULL
        ",
    );

    if let Some(fn_name) = &params.function_name {
        qb.push(" AND function_name = ");
        qb.push_bind(fn_name);
    }

    qb.push(
        r"
            GROUP BY dataset_name
            UNION ALL
            SELECT
                dataset_name,
                COUNT(*) AS count,
                MAX(updated_at) AS last_updated
            FROM tensorzero.json_datapoints
            WHERE staled_at IS NULL
        ",
    );

    if let Some(fn_name) = &params.function_name {
        qb.push(" AND function_name = ");
        qb.push_bind(fn_name);
    }

    qb.push(
        r"
            GROUP BY dataset_name
        )
        SELECT
            dataset_name,
            SUM(count)::INT AS count,
            MAX(last_updated) AS last_updated
        FROM unioned_datasets
        GROUP BY dataset_name
        ORDER BY MAX(last_updated) DESC, dataset_name ASC
        ",
    );

    if let Some(limit) = params.limit {
        qb.push(" LIMIT ");
        qb.push_bind(limit as i32);
    }

    if let Some(offset) = params.offset {
        qb.push(" OFFSET ");
        qb.push_bind(offset as i32);
    }

    qb
}

/// Builds a query to count datapoints for a dataset.
fn build_count_datapoints_query(
    dataset_name: &str,
    function_name: Option<&str>,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        SELECT (
            (SELECT COUNT(*) FROM tensorzero.chat_datapoints
             WHERE dataset_name = ",
    );
    qb.push_bind(dataset_name);
    qb.push(" AND staled_at IS NULL");

    if let Some(fn_name) = function_name {
        qb.push(" AND function_name = ");
        qb.push_bind(fn_name);
    }

    qb.push(
        r") +
            (SELECT COUNT(*) FROM tensorzero.json_datapoints
             WHERE dataset_name = ",
    );
    qb.push_bind(dataset_name);
    qb.push(" AND staled_at IS NULL");

    if let Some(fn_name) = function_name {
        qb.push(" AND function_name = ");
        qb.push_bind(fn_name);
    }

    qb.push(")) AS total");

    qb
}

/// Builds a UNION ALL query to select datapoints from both chat and json tables.
/// This pushes sorting and pagination to the database.
fn build_datapoints_union_query(
    params: &GetDatapointsParams,
) -> Result<QueryBuilder<sqlx::Postgres>, Error> {
    // Build ORDER BY clause string for reuse in inner and outer queries
    let order_by_clause = build_order_by_clause_string(params.order_by.as_ref());

    // Inner limit is (limit + offset) to ensure we fetch enough rows before applying outer offset
    let inner_limit = (params.limit + params.offset) as i64;

    // Build the chat subquery
    let mut qb = QueryBuilder::new(
        r"
        SELECT * FROM (
            (SELECT
                'chat'::text as datapoint_type,
                id, dataset_name, function_name, episode_id, input, output,
                dynamic_tools, dynamic_provider_tools, allowed_tools, tool_choice, parallel_tool_calls,
                NULL::jsonb as output_schema,
                tags, is_custom, source_inference_id, name, snapshot_hash, staled_at, updated_at
            FROM tensorzero.chat_datapoints
            WHERE TRUE
        ",
    );

    add_common_where_clauses(
        &mut qb,
        params.dataset_name.as_deref(),
        params.function_name.as_deref(),
        params.ids.as_deref(),
        params.allow_stale,
        params.filter.as_ref(),
        params.search_query_experimental.as_deref(),
        "input",
        "output",
    )?;

    qb.push(&order_by_clause);
    qb.push(" LIMIT ");
    qb.push_bind(inner_limit);

    // UNION ALL with json subquery
    qb.push(
        r")
            UNION ALL
            (SELECT
                'json'::text as datapoint_type,
                id, dataset_name, function_name, episode_id, input, output,
                NULL::jsonb as dynamic_tools, NULL::jsonb as dynamic_provider_tools,
                NULL::jsonb as allowed_tools, NULL::jsonb as tool_choice, NULL::boolean as parallel_tool_calls,
                output_schema,
                tags, is_custom, source_inference_id, name, snapshot_hash, staled_at, updated_at
            FROM tensorzero.json_datapoints
            WHERE TRUE
        ",
    );

    add_common_where_clauses(
        &mut qb,
        params.dataset_name.as_deref(),
        params.function_name.as_deref(),
        params.ids.as_deref(),
        params.allow_stale,
        params.filter.as_ref(),
        params.search_query_experimental.as_deref(),
        "input",
        "output",
    )?;

    qb.push(&order_by_clause);
    qb.push(" LIMIT ");
    qb.push_bind(inner_limit);

    // Close subqueries and add outer ORDER BY + LIMIT + OFFSET
    qb.push(
        r")
        ) AS combined
        ",
    );
    qb.push(&order_by_clause);
    qb.push(" LIMIT ");
    qb.push_bind(params.limit as i64);
    qb.push(" OFFSET ");
    qb.push_bind(params.offset as i64);

    Ok(qb)
}

/// Build ORDER BY clause as a string for use in UNION ALL queries.
fn build_order_by_clause_string(order_by: Option<&Vec<DatapointOrderBy>>) -> String {
    let Some(order_by_vec) = order_by else {
        return " ORDER BY updated_at DESC, id DESC".to_string();
    };

    if order_by_vec.is_empty() {
        return " ORDER BY updated_at DESC, id DESC".to_string();
    }

    let mut clauses = Vec::new();
    for order_spec in order_by_vec {
        let column = match order_spec.term {
            DatapointOrderByTerm::Timestamp => "updated_at",
            DatapointOrderByTerm::SearchRelevance => {
                // For Postgres, we don't have the same term frequency calculation as ClickHouse.
                // Fall back to updated_at for now.
                // TODO(#5691): Implement proper text search ranking in Postgres
                "updated_at"
            }
        };
        let direction = order_spec.direction.to_sql_direction();
        clauses.push(format!("{column} {direction}"));
    }

    // Always add id as tie-breaker for deterministic ordering
    clauses.push("id DESC".to_string());

    format!(" ORDER BY {}", clauses.join(", "))
}

/// Builds a query to delete (soft) chat datapoints.
fn build_delete_chat_datapoints_query(
    dataset_name: &str,
    datapoint_ids: Option<&[Uuid]>,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        UPDATE tensorzero.chat_datapoints
        SET staled_at = NOW(), updated_at = NOW()
        WHERE dataset_name = ",
    );
    qb.push_bind(dataset_name);
    qb.push(" AND staled_at IS NULL");

    if let Some(ids) = datapoint_ids {
        qb.push(" AND id = ANY(");
        qb.push_bind(ids.to_vec());
        qb.push(")");
    }

    qb
}

/// Builds a query to delete (soft) json datapoints.
fn build_delete_json_datapoints_query(
    dataset_name: &str,
    datapoint_ids: Option<&[Uuid]>,
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        UPDATE tensorzero.json_datapoints
        SET staled_at = NOW(), updated_at = NOW()
        WHERE dataset_name = ",
    );
    qb.push_bind(dataset_name);
    qb.push(" AND staled_at IS NULL");

    if let Some(ids) = datapoint_ids {
        qb.push(" AND id = ANY(");
        qb.push_bind(ids.to_vec());
        qb.push(")");
    }

    qb
}

/// Builds a query to fetch chat datapoints for cloning.
fn build_fetch_chat_datapoints_for_clone_query(
    source_ids: &[Uuid],
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        SELECT
            id, function_name, episode_id, input, output,
            dynamic_tools, dynamic_provider_tools, allowed_tools, tool_choice, parallel_tool_calls,
            tags, is_custom, source_inference_id, name, snapshot_hash
        FROM tensorzero.chat_datapoints
        WHERE id = ANY(",
    );
    qb.push_bind(source_ids.to_vec());
    qb.push(") AND staled_at IS NULL");

    qb
}

/// Builds a query to fetch json datapoints for cloning.
fn build_fetch_json_datapoints_for_clone_query(
    source_ids: &[Uuid],
) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        SELECT
            id, function_name, episode_id, input, output, output_schema,
            tags, is_custom, source_inference_id, name, snapshot_hash
        FROM tensorzero.json_datapoints
        WHERE id = ANY(",
    );
    qb.push_bind(source_ids.to_vec());
    qb.push(") AND staled_at IS NULL");

    qb
}

/// Builds a query to verify which IDs were created.
fn build_verify_created_ids_query(new_ids: &[Uuid]) -> QueryBuilder<sqlx::Postgres> {
    let mut qb = QueryBuilder::new(
        r"
        SELECT id FROM (
            SELECT id FROM tensorzero.chat_datapoints WHERE id = ANY(",
    );
    qb.push_bind(new_ids.to_vec());
    qb.push(") AND staled_at IS NULL UNION ALL SELECT id FROM tensorzero.json_datapoints WHERE id = ANY(");
    qb.push_bind(new_ids.to_vec());
    qb.push(") AND staled_at IS NULL) AS combined");

    qb
}

// =====================================================================
// Helper functions for inserting datapoints
// =====================================================================

async fn insert_chat_datapoints(
    pool: &PgPool,
    datapoints: &[&StoredChatInferenceDatapoint],
) -> Result<u64, Error> {
    if datapoints.is_empty() {
        return Ok(0);
    }

    for datapoint in datapoints {
        validate_dataset_name(&datapoint.dataset_name)?;
    }

    // Pre-compute timestamps from UUIDs for proper error propagation
    let timestamps: Vec<(Option<DateTime<Utc>>, DateTime<Utc>)> = datapoints
        .iter()
        .map(|dp| {
            let staled_at = dp
                .staled_at
                .as_ref()
                .map(|s| parse_clickhouse_datetime(s))
                .transpose()?;
            let created_at = uuid_to_datetime(dp.id)?;
            Ok((staled_at, created_at))
        })
        .collect::<Result<_, Error>>()?;

    let empty_tools: Vec<Tool> = vec![];
    let empty_provider_tools: Vec<ProviderTool> = vec![];
    let empty_tags: HashMap<String, String> = HashMap::new();

    let mut qb = QueryBuilder::new(
        r"
        INSERT INTO tensorzero.chat_datapoints (
            id, dataset_name, function_name, episode_id, input, output,
            dynamic_tools, dynamic_provider_tools, allowed_tools, tool_choice, parallel_tool_calls,
            tags, is_custom, source_inference_id, name, snapshot_hash, staled_at, created_at
        )
        ",
    );

    qb.push_values(
        datapoints.iter().zip(&timestamps),
        |mut b, (dp, (staled_at, created_at))| {
            let snapshot_hash_bytes = dp.snapshot_hash.as_ref().map(|h| h.as_bytes());
            let tool_params_ref = dp.tool_params.as_ref();

            b.push_bind(dp.id)
                .push_bind(&dp.dataset_name)
                .push_bind(&dp.function_name)
                .push_bind(dp.episode_id)
                .push_bind(Json::from(&dp.input))
                .push_bind(dp.output.as_ref().map(Json::from))
                .push_bind(Json::from(
                    tool_params_ref
                        .map(|tp| &tp.dynamic_tools)
                        .unwrap_or(&empty_tools),
                ))
                .push_bind(Json::from(
                    tool_params_ref
                        .map(|tp| &tp.dynamic_provider_tools)
                        .unwrap_or(&empty_provider_tools),
                ))
                .push_bind(tool_params_ref.map(|tp| Json::from(&tp.allowed_tools)))
                .push_bind(tool_params_ref.map(|tp| Json::from(&tp.tool_choice)))
                .push_bind(tool_params_ref.and_then(|tp| tp.parallel_tool_calls))
                .push_bind(Json::from(dp.tags.as_ref().unwrap_or(&empty_tags)))
                .push_bind(dp.is_custom)
                .push_bind(dp.source_inference_id)
                .push_bind(&dp.name)
                .push_bind(snapshot_hash_bytes)
                .push_bind(*staled_at)
                .push_bind(*created_at);
        },
    );

    qb.push(
        r"
        ON CONFLICT (id) DO UPDATE SET
            dataset_name = EXCLUDED.dataset_name,
            function_name = EXCLUDED.function_name,
            episode_id = EXCLUDED.episode_id,
            input = EXCLUDED.input,
            output = EXCLUDED.output,
            dynamic_tools = EXCLUDED.dynamic_tools,
            dynamic_provider_tools = EXCLUDED.dynamic_provider_tools,
            allowed_tools = EXCLUDED.allowed_tools,
            tool_choice = EXCLUDED.tool_choice,
            parallel_tool_calls = EXCLUDED.parallel_tool_calls,
            tags = EXCLUDED.tags,
            is_custom = EXCLUDED.is_custom,
            source_inference_id = EXCLUDED.source_inference_id,
            name = EXCLUDED.name,
            snapshot_hash = EXCLUDED.snapshot_hash,
            staled_at = EXCLUDED.staled_at,
            updated_at = NOW()
        ",
    );

    let result = qb.build().execute(pool).await?;

    Ok(result.rows_affected())
}

async fn insert_json_datapoints(
    pool: &PgPool,
    datapoints: &[&StoredJsonInferenceDatapoint],
) -> Result<u64, Error> {
    if datapoints.is_empty() {
        return Ok(0);
    }

    for datapoint in datapoints {
        validate_dataset_name(&datapoint.dataset_name)?;
    }

    // Pre-compute timestamps from UUIDs for proper error propagation
    let timestamps: Vec<(Option<DateTime<Utc>>, DateTime<Utc>)> = datapoints
        .iter()
        .map(|dp| {
            let staled_at = dp
                .staled_at
                .as_ref()
                .map(|s| parse_clickhouse_datetime(s))
                .transpose()?;
            let created_at = uuid_to_datetime(dp.id)?;
            Ok((staled_at, created_at))
        })
        .collect::<Result<_, Error>>()?;

    let empty_tags: HashMap<String, String> = HashMap::new();

    let mut qb = QueryBuilder::new(
        r"
        INSERT INTO tensorzero.json_datapoints (
            id, dataset_name, function_name, episode_id, input, output, output_schema,
            tags, is_custom, source_inference_id, name, snapshot_hash, staled_at, created_at
        )
        ",
    );

    qb.push_values(
        datapoints.iter().zip(&timestamps),
        |mut b, (dp, (staled_at, created_at))| {
            let snapshot_hash_bytes = dp.snapshot_hash.as_ref().map(|h| h.as_bytes());

            b.push_bind(dp.id)
                .push_bind(&dp.dataset_name)
                .push_bind(&dp.function_name)
                .push_bind(dp.episode_id)
                .push_bind(Json::from(&dp.input))
                .push_bind(dp.output.as_ref().map(Json::from))
                .push_bind(&dp.output_schema)
                .push_bind(Json::from(dp.tags.as_ref().unwrap_or(&empty_tags)))
                .push_bind(dp.is_custom)
                .push_bind(dp.source_inference_id)
                .push_bind(&dp.name)
                .push_bind(snapshot_hash_bytes)
                .push_bind(*staled_at)
                .push_bind(*created_at);
        },
    );

    qb.push(
        r"
        ON CONFLICT (id) DO UPDATE SET
            dataset_name = EXCLUDED.dataset_name,
            function_name = EXCLUDED.function_name,
            episode_id = EXCLUDED.episode_id,
            input = EXCLUDED.input,
            output = EXCLUDED.output,
            output_schema = EXCLUDED.output_schema,
            tags = EXCLUDED.tags,
            is_custom = EXCLUDED.is_custom,
            source_inference_id = EXCLUDED.source_inference_id,
            name = EXCLUDED.name,
            snapshot_hash = EXCLUDED.snapshot_hash,
            staled_at = EXCLUDED.staled_at,
            updated_at = NOW()
        ",
    );

    let result = qb.build().execute(pool).await?;

    Ok(result.rows_affected())
}

// =====================================================================
// FromRow implementations for datapoint types
// =====================================================================

/// Manual implementation of FromRow for StoredChatInferenceDatapoint.
/// This allows direct deserialization from Postgres rows.
impl<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> for StoredChatInferenceDatapoint {
    fn from_row(row: &'r sqlx::postgres::PgRow) -> Result<Self, sqlx::Error> {
        let input: Json<StoredInput> = row.try_get("input")?;
        let output: Option<Json<Vec<ContentBlockChatOutput>>> = row.try_get("output")?;
        let dynamic_tools: Json<Vec<Tool>> = row.try_get("dynamic_tools")?;
        let dynamic_provider_tools: Json<Vec<ProviderTool>> =
            row.try_get("dynamic_provider_tools")?;
        let allowed_tools: Option<Json<AllowedTools>> = row.try_get("allowed_tools")?;
        let tool_choice: Option<Json<ToolChoice>> = row.try_get("tool_choice")?;
        let parallel_tool_calls: Option<bool> = row.try_get("parallel_tool_calls")?;
        let tags: Json<HashMap<String, String>> = row.try_get("tags")?;
        let snapshot_hash_bytes: Option<Vec<u8>> = row.try_get("snapshot_hash")?;
        let staled_at: Option<DateTime<Utc>> = row.try_get("staled_at")?;
        let updated_at: DateTime<Utc> = row.try_get("updated_at")?;

        let tool_params = ToolCallConfigDatabaseInsert::from_stored_values(
            dynamic_tools.0,
            dynamic_provider_tools.0,
            allowed_tools.map(|v| v.0),
            tool_choice.map(|v| v.0),
            parallel_tool_calls,
        );

        let snapshot_hash = snapshot_hash_bytes.map(|b| SnapshotHash::from_bytes(&b));
        let tags = tags.0;

        Ok(StoredChatInferenceDatapoint {
            id: row.try_get("id")?,
            dataset_name: row.try_get("dataset_name")?,
            function_name: row.try_get("function_name")?,
            episode_id: row.try_get("episode_id")?,
            input: input.0,
            output: output.map(|v| v.0),
            tool_params,
            // TODO(shuyangli): Let's figure out whether we want to return empty maps as {} or skip
            tags: Some(tags),
            is_custom: row.try_get("is_custom")?,
            source_inference_id: row.try_get("source_inference_id")?,
            name: row.try_get("name")?,
            snapshot_hash,
            staled_at: staled_at.map(|dt| dt.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
            is_deleted: false,
            auxiliary: String::new(),
            updated_at: updated_at.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        })
    }
}

/// Manual implementation of FromRow for StoredJsonInferenceDatapoint.
/// This allows direct deserialization from Postgres rows.
impl<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> for StoredJsonInferenceDatapoint {
    fn from_row(row: &'r sqlx::postgres::PgRow) -> Result<Self, sqlx::Error> {
        let input: Json<StoredInput> = row.try_get("input")?;
        let output: Option<Json<JsonInferenceOutput>> = row.try_get("output")?;
        let output_schema: serde_json::Value = row.try_get("output_schema")?;
        let tags: Json<HashMap<String, String>> = row.try_get("tags")?;
        let snapshot_hash_bytes: Option<Vec<u8>> = row.try_get("snapshot_hash")?;
        let staled_at: Option<DateTime<Utc>> = row.try_get("staled_at")?;
        let updated_at: DateTime<Utc> = row.try_get("updated_at")?;

        let snapshot_hash = snapshot_hash_bytes.map(|b| SnapshotHash::from_bytes(&b));
        let tags = tags.0;

        Ok(StoredJsonInferenceDatapoint {
            id: row.try_get("id")?,
            dataset_name: row.try_get("dataset_name")?,
            function_name: row.try_get("function_name")?,
            episode_id: row.try_get("episode_id")?,
            input: input.0,
            output: output.map(|v| v.0),
            output_schema,
            // TODO(shuyangli): Let's figure out whether we want to return empty maps as {} or skip
            tags: Some(tags),
            is_custom: row.try_get("is_custom")?,
            source_inference_id: row.try_get("source_inference_id")?,
            name: row.try_get("name")?,
            snapshot_hash,
            staled_at: staled_at.map(|dt| dt.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
            is_deleted: false,
            auxiliary: String::new(),
            updated_at: updated_at.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        })
    }
}

/// Manual implementation of FromRow for StoredDatapoint to handle UNION ALL queries.
/// Uses the `datapoint_type` column ('chat' or 'json') to determine which variant to construct.
impl<'r> sqlx::FromRow<'r, PgRow> for StoredDatapoint {
    fn from_row(row: &'r PgRow) -> Result<Self, sqlx::Error> {
        let datapoint_type: String = row.try_get("datapoint_type")?;

        match datapoint_type.as_str() {
            "chat" => {
                let chat = StoredChatInferenceDatapoint::from_row(row)?;
                Ok(StoredDatapoint::Chat(chat))
            }
            "json" => {
                let json = StoredJsonInferenceDatapoint::from_row(row)?;
                Ok(StoredDatapoint::Json(json))
            }
            _ => Err(sqlx::Error::ColumnDecode {
                index: "datapoint_type".to_string(),
                source: Box::new(Error::new(ErrorDetails::PostgresResult {
                    result_type: "datapoint",
                    message: format!("Unknown datapoint type: {datapoint_type}"),
                })),
            }),
        }
    }
}

// =====================================================================
// Helper functions for querying datapoints
// =====================================================================

#[expect(clippy::too_many_arguments)]
fn add_common_where_clauses(
    qb: &mut QueryBuilder<sqlx::Postgres>,
    dataset_name: Option<&str>,
    function_name: Option<&str>,
    ids: Option<&[Uuid]>,
    allow_stale: bool,
    filter: Option<&DatapointFilter>,
    search_query: Option<&str>,
    input_column: &str,
    output_column: &str,
) -> Result<(), Error> {
    if let Some(dataset) = dataset_name {
        qb.push(" AND dataset_name = ");
        qb.push_bind(dataset.to_string());
    }

    if let Some(fn_name) = function_name {
        qb.push(" AND function_name = ");
        qb.push_bind(fn_name.to_string());
    }

    if let Some(ids_vec) = ids {
        qb.push(" AND id = ANY(");
        qb.push_bind(ids_vec.to_vec());
        qb.push(")");
    }

    if !allow_stale {
        qb.push(" AND staled_at IS NULL");
    }

    if let Some(f) = filter {
        qb.push(" AND ");
        add_filter_clause(qb, f)?;
    }

    if let Some(query) = search_query {
        // Case-insensitive substring search on input and output
        let json_escaped_query = json_double_escape_string_without_quotes(query)?;
        let search_pattern = format!("%{json_escaped_query}%");
        qb.push(" AND (");
        qb.push(input_column);
        qb.push("::TEXT ILIKE ");
        qb.push_bind(search_pattern.clone());
        qb.push(" OR ");
        qb.push(output_column);
        qb.push("::TEXT ILIKE ");
        qb.push_bind(search_pattern);
        qb.push(")");
    }

    Ok(())
}

fn add_filter_clause(
    qb: &mut QueryBuilder<sqlx::Postgres>,
    filter: &DatapointFilter,
) -> Result<(), Error> {
    match filter {
        DatapointFilter::Tag(TagFilter {
            key,
            value,
            comparison_operator,
        }) => {
            // tags->>key comparison value
            let op = match comparison_operator {
                TagComparisonOperator::Equal => "=",
                TagComparisonOperator::NotEqual => "<>",
            };
            qb.push("(tags ? ");
            qb.push_bind(key.clone());
            qb.push(" AND tags->>");
            qb.push_bind(key.clone());
            qb.push(" ");
            qb.push(op);
            qb.push(" ");
            qb.push_bind(value.clone());
            qb.push(")");
        }
        DatapointFilter::Time(TimeFilter {
            time,
            comparison_operator,
        }) => {
            let op = match comparison_operator {
                TimeComparisonOperator::LessThan => "<",
                TimeComparisonOperator::LessThanOrEqual => "<=",
                TimeComparisonOperator::Equal => "=",
                TimeComparisonOperator::GreaterThan => ">",
                TimeComparisonOperator::GreaterThanOrEqual => ">=",
                TimeComparisonOperator::NotEqual => "<>",
            };
            qb.push("updated_at ");
            qb.push(op);
            qb.push(" ");
            qb.push_bind(*time);
        }
        DatapointFilter::And { children } => {
            if children.is_empty() {
                qb.push("TRUE");
            } else {
                qb.push("(");
                for (i, child) in children.iter().enumerate() {
                    if i > 0 {
                        qb.push(" AND ");
                    }
                    add_filter_clause(qb, child)?;
                }
                qb.push(")");
            }
        }
        DatapointFilter::Or { children } => {
            if children.is_empty() {
                qb.push("FALSE");
            } else {
                qb.push("(");
                for (i, child) in children.iter().enumerate() {
                    if i > 0 {
                        qb.push(" OR ");
                    }
                    add_filter_clause(qb, child)?;
                }
                qb.push(")");
            }
        }
        DatapointFilter::Not { child } => {
            qb.push("NOT (");
            add_filter_clause(qb, child)?;
            qb.push(")");
        }
    }
    Ok(())
}

// =====================================================================
// Helper functions for deleting datapoints
// =====================================================================

async fn delete_chat_datapoints(
    pool: &PgPool,
    dataset_name: &str,
    datapoint_ids: Option<&[Uuid]>,
) -> Result<u64, Error> {
    let mut qb = build_delete_chat_datapoints_query(dataset_name, datapoint_ids);

    let result = qb.build().execute(pool).await?;

    Ok(result.rows_affected())
}

async fn delete_json_datapoints(
    pool: &PgPool,
    dataset_name: &str,
    datapoint_ids: Option<&[Uuid]>,
) -> Result<u64, Error> {
    let mut qb = build_delete_json_datapoints_query(dataset_name, datapoint_ids);

    let result = qb.build().execute(pool).await?;

    Ok(result.rows_affected())
}

// =====================================================================
// Helper functions for cloning datapoints
// =====================================================================

async fn clone_chat_datapoints(
    pool: &PgPool,
    target_dataset_name: &str,
    mappings: &[(Uuid, Uuid)],
) -> Result<(), Error> {
    if mappings.is_empty() {
        return Ok(());
    }

    let source_ids: Vec<Uuid> = mappings.iter().map(|(src, _)| *src).collect();
    let id_map: HashMap<Uuid, Uuid> = mappings.iter().copied().collect();

    // Fetch source datapoints
    let mut qb = build_fetch_chat_datapoints_for_clone_query(&source_ids);

    let rows = qb.build().fetch_all(pool).await?;

    if rows.is_empty() {
        return Ok(());
    }

    // Pre-compute new IDs and their timestamps, propagating errors
    let rows_with_mappings: Vec<(&PgRow, Uuid, DateTime<Utc>)> = rows
        .iter()
        .map(|row| {
            let source_id: Uuid = row.get("id");
            let new_id = id_map.get(&source_id).copied().ok_or_else(|| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!(
                        "Missing ID mapping for chat datapoint clone: source_id={source_id}"
                    ),
                })
            })?;
            let created_at = uuid_to_datetime(new_id)?;
            Ok((row, new_id, created_at))
        })
        .collect::<Result<Vec<_>, Error>>()?;

    // Insert cloned datapoints
    let mut insert_qb = QueryBuilder::new(
        r"
        INSERT INTO tensorzero.chat_datapoints (
            id, dataset_name, function_name, episode_id, input, output,
            dynamic_tools, dynamic_provider_tools, allowed_tools, tool_choice, parallel_tool_calls,
            tags, is_custom, source_inference_id, name, snapshot_hash, created_at
        )
        ",
    );

    insert_qb.push_values(&rows_with_mappings, |mut b, (row, new_id, created_at)| {
        b.push_bind(*new_id)
            .push_bind(target_dataset_name)
            .push_bind(row.get::<String, _>("function_name"))
            .push_bind(row.get::<Option<Uuid>, _>("episode_id"))
            .push_bind(row.get::<serde_json::Value, _>("input"))
            .push_bind(row.get::<Option<serde_json::Value>, _>("output"))
            .push_bind(row.get::<serde_json::Value, _>("dynamic_tools"))
            .push_bind(row.get::<serde_json::Value, _>("dynamic_provider_tools"))
            .push_bind(row.get::<Option<serde_json::Value>, _>("allowed_tools"))
            .push_bind(row.get::<Option<serde_json::Value>, _>("tool_choice"))
            .push_bind(row.get::<Option<bool>, _>("parallel_tool_calls"))
            .push_bind(row.get::<serde_json::Value, _>("tags"))
            .push_bind(row.get::<bool, _>("is_custom"))
            .push_bind(row.get::<Option<Uuid>, _>("source_inference_id"))
            .push_bind(row.get::<Option<String>, _>("name"))
            .push_bind(row.get::<Option<Vec<u8>>, _>("snapshot_hash"))
            .push_bind(*created_at);
    });

    insert_qb.build().execute(pool).await?;

    Ok(())
}

async fn clone_json_datapoints(
    pool: &PgPool,
    target_dataset_name: &str,
    mappings: &[(Uuid, Uuid)],
) -> Result<(), Error> {
    if mappings.is_empty() {
        return Ok(());
    }

    let source_ids: Vec<Uuid> = mappings.iter().map(|(src, _)| *src).collect();
    let id_map: HashMap<Uuid, Uuid> = mappings.iter().copied().collect();

    // Fetch source datapoints
    let mut qb = build_fetch_json_datapoints_for_clone_query(&source_ids);

    let rows = qb.build().fetch_all(pool).await?;

    if rows.is_empty() {
        return Ok(());
    }

    // Pre-compute new IDs and their timestamps, propagating errors
    let rows_with_mappings: Vec<(&PgRow, Uuid, DateTime<Utc>)> = rows
        .iter()
        .map(|row| {
            let source_id: Uuid = row.get("id");
            let new_id = id_map.get(&source_id).copied().ok_or_else(|| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!(
                        "Missing ID mapping for json datapoint clone: source_id={source_id}"
                    ),
                })
            })?;
            let created_at = uuid_to_datetime(new_id)?;
            Ok((row, new_id, created_at))
        })
        .collect::<Result<Vec<_>, Error>>()?;

    // Insert cloned datapoints
    let mut insert_qb = QueryBuilder::new(
        r"
        INSERT INTO tensorzero.json_datapoints (
            id, dataset_name, function_name, episode_id, input, output, output_schema,
            tags, is_custom, source_inference_id, name, snapshot_hash, created_at
        )
        ",
    );

    insert_qb.push_values(&rows_with_mappings, |mut b, (row, new_id, created_at)| {
        b.push_bind(*new_id)
            .push_bind(target_dataset_name)
            .push_bind(row.get::<String, _>("function_name"))
            .push_bind(row.get::<Option<Uuid>, _>("episode_id"))
            .push_bind(row.get::<serde_json::Value, _>("input"))
            .push_bind(row.get::<Option<serde_json::Value>, _>("output"))
            .push_bind(row.get::<serde_json::Value, _>("output_schema"))
            .push_bind(row.get::<serde_json::Value, _>("tags"))
            .push_bind(row.get::<bool, _>("is_custom"))
            .push_bind(row.get::<Option<Uuid>, _>("source_inference_id"))
            .push_bind(row.get::<Option<String>, _>("name"))
            .push_bind(row.get::<Option<Vec<u8>>, _>("snapshot_hash"))
            .push_bind(*created_at);
    });

    insert_qb.build().execute(pool).await?;

    Ok(())
}

async fn verify_created_ids(pool: &PgPool, new_ids: &[Uuid]) -> Result<HashSet<Uuid>, Error> {
    if new_ids.is_empty() {
        return Ok(HashSet::new());
    }

    let mut qb = build_verify_created_ids_query(new_ids);

    let rows = qb.build().fetch_all(pool).await?;

    Ok(rows.into_iter().map(|row| row.get("id")).collect())
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test_helpers::assert_query_equals;
    use crate::endpoints::shared_types::OrderDirection;

    #[test]
    fn test_build_get_dataset_metadata_query_no_filters() {
        let params = GetDatasetMetadataParams {
            function_name: None,
            limit: None,
            offset: None,
        };

        let qb = build_get_dataset_metadata_query(&params);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH unioned_datasets AS (
                SELECT
                    dataset_name,
                    COUNT(*) AS count,
                    MAX(updated_at) AS last_updated
                FROM tensorzero.chat_datapoints
                WHERE staled_at IS NULL
                GROUP BY dataset_name
                UNION ALL
                SELECT
                    dataset_name,
                    COUNT(*) AS count,
                    MAX(updated_at) AS last_updated
                FROM tensorzero.json_datapoints
                WHERE staled_at IS NULL
                GROUP BY dataset_name
            )
            SELECT
                dataset_name,
                SUM(count)::INT AS count,
                MAX(last_updated) AS last_updated
            FROM unioned_datasets
            GROUP BY dataset_name
            ORDER BY MAX(last_updated) DESC, dataset_name ASC
            ",
        );
    }

    #[test]
    fn test_build_get_dataset_metadata_query_with_function_name() {
        let params = GetDatasetMetadataParams {
            function_name: Some("test_function".to_string()),
            limit: None,
            offset: None,
        };

        let qb = build_get_dataset_metadata_query(&params);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH unioned_datasets AS (
                SELECT
                    dataset_name,
                    COUNT(*) AS count,
                    MAX(updated_at) AS last_updated
                FROM tensorzero.chat_datapoints
                WHERE staled_at IS NULL
                AND function_name = $1
                GROUP BY dataset_name
                UNION ALL
                SELECT
                    dataset_name,
                    COUNT(*) AS count,
                    MAX(updated_at) AS last_updated
                FROM tensorzero.json_datapoints
                WHERE staled_at IS NULL
                AND function_name = $2
                GROUP BY dataset_name
            )
            SELECT
                dataset_name,
                SUM(count)::INT AS count,
                MAX(last_updated) AS last_updated
            FROM unioned_datasets
            GROUP BY dataset_name
            ORDER BY MAX(last_updated) DESC, dataset_name ASC
            ",
        );
    }

    #[test]
    fn test_build_get_dataset_metadata_query_with_limit_and_offset() {
        let params = GetDatasetMetadataParams {
            function_name: None,
            limit: Some(10),
            offset: Some(5),
        };

        let qb = build_get_dataset_metadata_query(&params);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            WITH unioned_datasets AS (
                SELECT
                    dataset_name,
                    COUNT(*) AS count,
                    MAX(updated_at) AS last_updated
                FROM tensorzero.chat_datapoints
                WHERE staled_at IS NULL
                GROUP BY dataset_name
                UNION ALL
                SELECT
                    dataset_name,
                    COUNT(*) AS count,
                    MAX(updated_at) AS last_updated
                FROM tensorzero.json_datapoints
                WHERE staled_at IS NULL
                GROUP BY dataset_name
            )
            SELECT
                dataset_name,
                SUM(count)::INT AS count,
                MAX(last_updated) AS last_updated
            FROM unioned_datasets
            GROUP BY dataset_name
            ORDER BY MAX(last_updated) DESC, dataset_name ASC
            LIMIT $1 OFFSET $2
            ",
        );
    }

    #[test]
    fn test_build_count_datapoints_query_no_function_name() {
        let qb = build_count_datapoints_query("my_dataset", None);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT (
                (SELECT COUNT(*) FROM tensorzero.chat_datapoints
                 WHERE dataset_name = $1 AND staled_at IS NULL) +
                (SELECT COUNT(*) FROM tensorzero.json_datapoints
                 WHERE dataset_name = $2 AND staled_at IS NULL)) AS total
            ",
        );
    }

    #[test]
    fn test_build_count_datapoints_query_with_function_name() {
        let qb = build_count_datapoints_query("my_dataset", Some("my_function"));
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT (
                (SELECT COUNT(*) FROM tensorzero.chat_datapoints
                 WHERE dataset_name = $1 AND staled_at IS NULL AND function_name = $2) +
                (SELECT COUNT(*) FROM tensorzero.json_datapoints
                 WHERE dataset_name = $3 AND staled_at IS NULL AND function_name = $4)) AS total
            ",
        );
    }

    #[test]
    fn test_build_datapoints_union_query_basic() {
        let params = GetDatapointsParams {
            dataset_name: Some("my_dataset".to_string()),
            function_name: None,
            ids: None,
            limit: 10,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        };

        let qb = build_datapoints_union_query(&params).unwrap();
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT * FROM (
                (SELECT
                    'chat'::text as datapoint_type,
                    id, dataset_name, function_name, episode_id, input, output,
                    dynamic_tools, dynamic_provider_tools, allowed_tools, tool_choice, parallel_tool_calls,
                    NULL::jsonb as output_schema,
                    tags, is_custom, source_inference_id, name, snapshot_hash, staled_at, updated_at
                FROM tensorzero.chat_datapoints
                WHERE TRUE
                AND dataset_name = $1 AND staled_at IS NULL
                ORDER BY updated_at DESC, id DESC LIMIT $2)
                UNION ALL
                (SELECT
                    'json'::text as datapoint_type,
                    id, dataset_name, function_name, episode_id, input, output,
                    NULL::jsonb as dynamic_tools, NULL::jsonb as dynamic_provider_tools,
                    NULL::jsonb as allowed_tools, NULL::jsonb as tool_choice, NULL::boolean as parallel_tool_calls,
                    output_schema,
                    tags, is_custom, source_inference_id, name, snapshot_hash, staled_at, updated_at
                FROM tensorzero.json_datapoints
                WHERE TRUE
                AND dataset_name = $3 AND staled_at IS NULL
                ORDER BY updated_at DESC, id DESC LIMIT $4)
            ) AS combined
            ORDER BY updated_at DESC, id DESC LIMIT $5 OFFSET $6
            ",
        );
    }

    #[test]
    fn test_build_datapoints_union_query_with_all_filters() {
        let ids = vec![Uuid::now_v7()];
        let order_by = vec![DatapointOrderBy {
            term: DatapointOrderByTerm::Timestamp,
            direction: OrderDirection::Asc,
        }];

        let params = GetDatapointsParams {
            dataset_name: Some("my_dataset".to_string()),
            function_name: Some("my_function".to_string()),
            ids: Some(ids),
            limit: 10,
            offset: 5,
            allow_stale: true,
            filter: None,
            order_by: Some(order_by),
            search_query_experimental: Some("search term".to_string()),
        };

        let qb = build_datapoints_union_query(&params).unwrap();
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT * FROM (
                (SELECT
                    'chat'::text as datapoint_type,
                    id, dataset_name, function_name, episode_id, input, output,
                    dynamic_tools, dynamic_provider_tools, allowed_tools, tool_choice, parallel_tool_calls,
                    NULL::jsonb as output_schema,
                    tags, is_custom, source_inference_id, name, snapshot_hash, staled_at, updated_at
                FROM tensorzero.chat_datapoints
                WHERE TRUE
                AND dataset_name = $1 AND function_name = $2 AND id = ANY($3)
                AND (input::TEXT ILIKE $4 OR output::TEXT ILIKE $5)
                ORDER BY updated_at ASC, id DESC LIMIT $6)
                UNION ALL
                (SELECT
                    'json'::text as datapoint_type,
                    id, dataset_name, function_name, episode_id, input, output,
                    NULL::jsonb as dynamic_tools, NULL::jsonb as dynamic_provider_tools,
                    NULL::jsonb as allowed_tools, NULL::jsonb as tool_choice, NULL::boolean as parallel_tool_calls,
                    output_schema,
                    tags, is_custom, source_inference_id, name, snapshot_hash, staled_at, updated_at
                FROM tensorzero.json_datapoints
                WHERE TRUE
                AND dataset_name = $7 AND function_name = $8 AND id = ANY($9)
                AND (input::TEXT ILIKE $10 OR output::TEXT ILIKE $11)
                ORDER BY updated_at ASC, id DESC LIMIT $12)
            ) AS combined
            ORDER BY updated_at ASC, id DESC LIMIT $13 OFFSET $14
            ",
        );
    }

    #[test]
    fn test_build_delete_chat_datapoints_query_all() {
        let qb = build_delete_chat_datapoints_query("my_dataset", None);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            UPDATE tensorzero.chat_datapoints
            SET staled_at = NOW(), updated_at = NOW()
            WHERE dataset_name = $1 AND staled_at IS NULL
            ",
        );
    }

    #[test]
    fn test_build_delete_chat_datapoints_query_specific_ids() {
        let ids = vec![Uuid::now_v7()];
        let qb = build_delete_chat_datapoints_query("my_dataset", Some(&ids));
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            UPDATE tensorzero.chat_datapoints
            SET staled_at = NOW(), updated_at = NOW()
            WHERE dataset_name = $1 AND staled_at IS NULL AND id = ANY($2)
            ",
        );
    }

    #[test]
    fn test_build_delete_json_datapoints_query_all() {
        let qb = build_delete_json_datapoints_query("my_dataset", None);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            UPDATE tensorzero.json_datapoints
            SET staled_at = NOW(), updated_at = NOW()
            WHERE dataset_name = $1 AND staled_at IS NULL
            ",
        );
    }

    #[test]
    fn test_build_delete_json_datapoints_query_specific_ids() {
        let ids = vec![Uuid::now_v7()];
        let qb = build_delete_json_datapoints_query("my_dataset", Some(&ids));
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            UPDATE tensorzero.json_datapoints
            SET staled_at = NOW(), updated_at = NOW()
            WHERE dataset_name = $1 AND staled_at IS NULL AND id = ANY($2)
            ",
        );
    }

    #[test]
    fn test_build_fetch_chat_datapoints_for_clone_query() {
        let source_ids = vec![Uuid::now_v7()];
        let qb = build_fetch_chat_datapoints_for_clone_query(&source_ids);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                id, function_name, episode_id, input, output,
                dynamic_tools, dynamic_provider_tools, allowed_tools, tool_choice, parallel_tool_calls,
                tags, is_custom, source_inference_id, name, snapshot_hash
            FROM tensorzero.chat_datapoints
            WHERE id = ANY($1) AND staled_at IS NULL
            ",
        );
    }

    #[test]
    fn test_build_fetch_json_datapoints_for_clone_query() {
        let source_ids = vec![Uuid::now_v7()];
        let qb = build_fetch_json_datapoints_for_clone_query(&source_ids);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT
                id, function_name, episode_id, input, output, output_schema,
                tags, is_custom, source_inference_id, name, snapshot_hash
            FROM tensorzero.json_datapoints
            WHERE id = ANY($1) AND staled_at IS NULL
            ",
        );
    }

    #[test]
    fn test_build_verify_created_ids_query() {
        let new_ids = vec![Uuid::now_v7()];
        let qb = build_verify_created_ids_query(&new_ids);
        let sql_str = qb.sql();
        let sql = sql_str.as_str();

        assert_query_equals(
            sql,
            r"
            SELECT id FROM (
                SELECT id FROM tensorzero.chat_datapoints WHERE id = ANY($1) AND staled_at IS NULL
                UNION ALL SELECT id FROM tensorzero.json_datapoints WHERE id = ANY($2) AND staled_at IS NULL) AS combined
            ",
        );
    }

    #[test]
    fn test_parse_clickhouse_datetime() {
        // Test parsing ClickHouse datetime format with microseconds
        let result = parse_clickhouse_datetime("2025-01-15 12:34:56.123456").unwrap();
        assert_eq!(
            result.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2025-01-15 12:34:56"
        );

        // Test parsing without microseconds (6 zeros)
        let result = parse_clickhouse_datetime("2025-01-15 12:34:56.000000").unwrap();
        assert_eq!(
            result.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2025-01-15 12:34:56"
        );

        // Test that RFC 3339 format fails (as expected - we only support ClickHouse format)
        let result = parse_clickhouse_datetime("2025-01-15T12:34:56Z");
        assert!(result.is_err(), "RFC 3339 format should fail");
    }
}
