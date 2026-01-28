//! Dataset queries for Postgres.
//!
//! This module implements both read and write operations for datapoint tables in Postgres.

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::{PgPool, QueryBuilder, Row};
use uuid::Uuid;

use crate::config::snapshot::SnapshotHash;
use crate::db::clickhouse::query_builder::{
    DatapointFilter, TagComparisonOperator, TagFilter, TimeComparisonOperator, TimeFilter,
};
use crate::db::datasets::{
    DatasetMetadata, DatasetQueries, GetDatapointParams, GetDatapointsParams,
    GetDatasetMetadataParams,
};
use crate::db::stored_datapoint::{
    StoredChatInferenceDatapoint, StoredDatapoint, StoredJsonInferenceDatapoint,
};
use crate::endpoints::datasets::v1::types::{DatapointOrderBy, DatapointOrderByTerm};
use crate::endpoints::datasets::validate_dataset_name;
use crate::endpoints::shared_types::OrderDirection;
use crate::error::{Error, ErrorDetails};

use super::PostgresConnectionInfo;

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

        let rows = qb.build().fetch_all(pool).await.map_err(Error::from)?;

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

        qb.push(")::BIGINT AS total");

        let row = qb.build().fetch_one(pool).await.map_err(Error::from)?;
        let total: i64 = row.get("total");

        Ok(total as u64)
    }

    async fn get_datapoint(&self, params: &GetDatapointParams) -> Result<StoredDatapoint, Error> {
        const DEFAULT_ALLOW_STALE_IN_GET_DATAPOINT: bool = false;
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

        let GetDatapointsParams {
            dataset_name,
            function_name,
            ids,
            limit,
            offset,
            allow_stale,
            filter,
            order_by,
            search_query_experimental,
        } = params;

        // If neither IDs nor dataset are provided, reject the query.
        if dataset_name.is_none() && ids.is_none() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "At least one of dataset_name or ids must be provided".to_string(),
            }));
        }

        // If IDs are provided, they must not be empty.
        if let Some(ids_vec) = ids
            && ids_vec.is_empty()
        {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "ids must not be an empty list".to_string(),
            }));
        }

        // Validate that if SearchRelevance is used, search_query_experimental must be present
        if let Some(order_by_vec) = order_by {
            for order_spec in order_by_vec {
                if matches!(order_spec.term, DatapointOrderByTerm::SearchRelevance)
                    && search_query_experimental.is_none()
                {
                    return Err(Error::new(ErrorDetails::InvalidRequest {
                        message:
                            "OrderBy::SearchRelevance requires search_query_experimental to be provided"
                                .to_string(),
                    }));
                }
            }
        }

        // Query both tables in parallel
        let (chat_datapoints, json_datapoints) = tokio::join!(
            query_chat_datapoints(
                pool,
                dataset_name.as_deref(),
                function_name.as_deref(),
                ids.as_deref(),
                *limit,
                *offset,
                *allow_stale,
                filter.as_ref(),
                order_by.as_ref(),
                search_query_experimental.as_deref(),
            ),
            query_json_datapoints(
                pool,
                dataset_name.as_deref(),
                function_name.as_deref(),
                ids.as_deref(),
                *limit,
                *offset,
                *allow_stale,
                filter.as_ref(),
                order_by.as_ref(),
                search_query_experimental.as_deref(),
            ),
        );

        let mut chat_datapoints = chat_datapoints?;
        let mut json_datapoints = json_datapoints?;

        // Merge results
        let mut all_datapoints: Vec<StoredDatapoint> =
            Vec::with_capacity(chat_datapoints.len() + json_datapoints.len());
        all_datapoints.append(&mut chat_datapoints);
        all_datapoints.append(&mut json_datapoints);

        // Sort by the ordering criteria (default: updated_at DESC, id DESC)
        sort_datapoints(&mut all_datapoints, order_by.as_ref());

        // Apply pagination after merge
        let start = (*offset as usize).min(all_datapoints.len());
        let end = (start + *limit as usize).min(all_datapoints.len());
        let result = all_datapoints[start..end].to_vec();

        Ok(result)
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
    ) -> Result<Vec<Option<Uuid>>, Error> {
        let pool = self.get_pool_result()?;

        if source_datapoint_ids.is_empty() {
            return Ok(vec![]);
        }

        // Generate all mappings from source to target IDs
        let mappings: Vec<(Uuid, Uuid)> = source_datapoint_ids
            .iter()
            .map(|id| (*id, Uuid::now_v7()))
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

    let mut qb = QueryBuilder::new(
        r"
        INSERT INTO tensorzero.chat_datapoints (
            id, dataset_name, function_name, episode_id, input, output, tool_params,
            tags, is_custom, source_inference_id, name, snapshot_hash, staled_at
        )
        ",
    );

    qb.push_values(datapoints, |mut b, dp| {
        let input_json = serde_json::to_value(&dp.input).unwrap_or_default();
        let output_json = dp
            .output
            .as_ref()
            .map(|o| serde_json::to_value(o).unwrap_or_default());
        let tool_params_json = dp
            .tool_params
            .as_ref()
            .map(|t| serde_json::to_value(t).unwrap_or_default());
        let tags_json =
            serde_json::to_value(dp.tags.clone().unwrap_or_default()).unwrap_or_default();
        let snapshot_hash_bytes = dp.snapshot_hash.as_ref().map(|h| h.as_bytes());

        b.push_bind(dp.id)
            .push_bind(&dp.dataset_name)
            .push_bind(&dp.function_name)
            .push_bind(dp.episode_id)
            .push_bind(input_json)
            .push_bind(output_json)
            .push_bind(tool_params_json)
            .push_bind(tags_json)
            .push_bind(dp.is_custom)
            .push_bind(dp.source_inference_id)
            .push_bind(&dp.name)
            .push_bind(snapshot_hash_bytes)
            .push_bind(
                dp.staled_at
                    .as_ref()
                    .and_then(|s| s.parse::<DateTime<Utc>>().ok()),
            );
    });

    qb.push(
        r"
        ON CONFLICT (id) DO UPDATE SET
            dataset_name = EXCLUDED.dataset_name,
            function_name = EXCLUDED.function_name,
            episode_id = EXCLUDED.episode_id,
            input = EXCLUDED.input,
            output = EXCLUDED.output,
            tool_params = EXCLUDED.tool_params,
            tags = EXCLUDED.tags,
            is_custom = EXCLUDED.is_custom,
            source_inference_id = EXCLUDED.source_inference_id,
            name = EXCLUDED.name,
            snapshot_hash = EXCLUDED.snapshot_hash,
            staled_at = EXCLUDED.staled_at,
            updated_at = NOW()
        ",
    );

    let result = qb.build().execute(pool).await.map_err(|e| {
        Error::new(ErrorDetails::PostgresConnection {
            message: format!("Failed to insert chat datapoints: {e}"),
        })
    })?;

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

    let mut qb = QueryBuilder::new(
        r"
        INSERT INTO tensorzero.json_datapoints (
            id, dataset_name, function_name, episode_id, input, output, output_schema,
            tags, is_custom, source_inference_id, name, snapshot_hash, staled_at
        )
        ",
    );

    qb.push_values(datapoints, |mut b, dp| {
        let input_json = serde_json::to_value(&dp.input).unwrap_or_default();
        let output_json = dp
            .output
            .as_ref()
            .map(|o| serde_json::to_value(o).unwrap_or_default());
        let tags_json =
            serde_json::to_value(dp.tags.clone().unwrap_or_default()).unwrap_or_default();
        let snapshot_hash_bytes = dp.snapshot_hash.as_ref().map(|h| h.as_bytes());

        b.push_bind(dp.id)
            .push_bind(&dp.dataset_name)
            .push_bind(&dp.function_name)
            .push_bind(dp.episode_id)
            .push_bind(input_json)
            .push_bind(output_json)
            .push_bind(&dp.output_schema)
            .push_bind(tags_json)
            .push_bind(dp.is_custom)
            .push_bind(dp.source_inference_id)
            .push_bind(&dp.name)
            .push_bind(snapshot_hash_bytes)
            .push_bind(
                dp.staled_at
                    .as_ref()
                    .and_then(|s| s.parse::<DateTime<Utc>>().ok()),
            );
    });

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

    let result = qb.build().execute(pool).await.map_err(|e| {
        Error::new(ErrorDetails::PostgresConnection {
            message: format!("Failed to insert json datapoints: {e}"),
        })
    })?;

    Ok(result.rows_affected())
}

// =====================================================================
// Helper functions for querying datapoints
// =====================================================================

#[expect(clippy::too_many_arguments)]
async fn query_chat_datapoints(
    pool: &PgPool,
    dataset_name: Option<&str>,
    function_name: Option<&str>,
    ids: Option<&[Uuid]>,
    limit: u32,
    offset: u32,
    allow_stale: bool,
    filter: Option<&DatapointFilter>,
    order_by: Option<&Vec<DatapointOrderBy>>,
    search_query: Option<&str>,
) -> Result<Vec<StoredDatapoint>, Error> {
    let subquery_limit = (limit + offset) as i64;

    let mut qb = QueryBuilder::new(
        r"
        SELECT
            id, dataset_name, function_name, episode_id, input, output, tool_params,
            tags, is_custom, source_inference_id, name, snapshot_hash, staled_at, updated_at
        FROM tensorzero.chat_datapoints
        WHERE TRUE
        ",
    );

    add_common_where_clauses(
        &mut qb,
        dataset_name,
        function_name,
        ids,
        allow_stale,
        filter,
        search_query,
        "input",
        "output",
    )?;

    add_order_by_clause(&mut qb, order_by, search_query);

    qb.push(" LIMIT ");
    qb.push_bind(subquery_limit);

    let rows = qb.build().fetch_all(pool).await.map_err(Error::from)?;

    rows.into_iter()
        .map(|row| {
            let input_json: serde_json::Value = row.get("input");
            let output_json: Option<serde_json::Value> = row.get("output");
            let tool_params_json: Option<serde_json::Value> = row.get("tool_params");
            let tags_json: serde_json::Value = row.get("tags");
            let snapshot_hash_bytes: Option<Vec<u8>> = row.get("snapshot_hash");
            let staled_at: Option<DateTime<Utc>> = row.get("staled_at");
            let updated_at: DateTime<Utc> = row.get("updated_at");

            let input = serde_json::from_value(input_json).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to deserialize chat datapoint input: {e}"),
                })
            })?;

            let output = output_json
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!("Failed to deserialize chat datapoint output: {e}"),
                    })
                })?;

            let tool_params = tool_params_json
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!("Failed to deserialize chat datapoint tool_params: {e}"),
                    })
                })?;

            let tags: HashMap<String, String> =
                serde_json::from_value(tags_json).unwrap_or_default();

            let snapshot_hash = snapshot_hash_bytes.map(|b| SnapshotHash::from_bytes(&b));

            Ok(StoredDatapoint::Chat(StoredChatInferenceDatapoint {
                id: row.get("id"),
                dataset_name: row.get("dataset_name"),
                function_name: row.get("function_name"),
                episode_id: row.get("episode_id"),
                input,
                output,
                tool_params,
                tags: if tags.is_empty() { None } else { Some(tags) },
                is_custom: row.get("is_custom"),
                source_inference_id: row.get("source_inference_id"),
                name: row.get("name"),
                snapshot_hash,
                staled_at: staled_at.map(|dt| dt.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
                is_deleted: false,
                auxiliary: String::new(),
                updated_at: updated_at.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
            }))
        })
        .collect()
}

#[expect(clippy::too_many_arguments)]
async fn query_json_datapoints(
    pool: &PgPool,
    dataset_name: Option<&str>,
    function_name: Option<&str>,
    ids: Option<&[Uuid]>,
    limit: u32,
    offset: u32,
    allow_stale: bool,
    filter: Option<&DatapointFilter>,
    order_by: Option<&Vec<DatapointOrderBy>>,
    search_query: Option<&str>,
) -> Result<Vec<StoredDatapoint>, Error> {
    let subquery_limit = (limit + offset) as i64;

    let mut qb = QueryBuilder::new(
        r"
        SELECT
            id, dataset_name, function_name, episode_id, input, output, output_schema,
            tags, is_custom, source_inference_id, name, snapshot_hash, staled_at, updated_at
        FROM tensorzero.json_datapoints
        WHERE TRUE
        ",
    );

    add_common_where_clauses(
        &mut qb,
        dataset_name,
        function_name,
        ids,
        allow_stale,
        filter,
        search_query,
        "input",
        "output",
    )?;

    add_order_by_clause(&mut qb, order_by, search_query);

    qb.push(" LIMIT ");
    qb.push_bind(subquery_limit);

    let rows = qb.build().fetch_all(pool).await.map_err(Error::from)?;

    rows.into_iter()
        .map(|row| {
            let input_json: serde_json::Value = row.get("input");
            let output_json: Option<serde_json::Value> = row.get("output");
            let output_schema: serde_json::Value = row.get("output_schema");
            let tags_json: serde_json::Value = row.get("tags");
            let snapshot_hash_bytes: Option<Vec<u8>> = row.get("snapshot_hash");
            let staled_at: Option<DateTime<Utc>> = row.get("staled_at");
            let updated_at: DateTime<Utc> = row.get("updated_at");

            let input = serde_json::from_value(input_json).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to deserialize json datapoint input: {e}"),
                })
            })?;

            let output = output_json
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!("Failed to deserialize json datapoint output: {e}"),
                    })
                })?;

            let tags: HashMap<String, String> =
                serde_json::from_value(tags_json).unwrap_or_default();

            let snapshot_hash = snapshot_hash_bytes.map(|b| SnapshotHash::from_bytes(&b));

            Ok(StoredDatapoint::Json(StoredJsonInferenceDatapoint {
                id: row.get("id"),
                dataset_name: row.get("dataset_name"),
                function_name: row.get("function_name"),
                episode_id: row.get("episode_id"),
                input,
                output,
                output_schema,
                tags: if tags.is_empty() { None } else { Some(tags) },
                is_custom: row.get("is_custom"),
                source_inference_id: row.get("source_inference_id"),
                name: row.get("name"),
                snapshot_hash,
                staled_at: staled_at.map(|dt| dt.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
                is_deleted: false,
                auxiliary: String::new(),
                updated_at: updated_at.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
            }))
        })
        .collect()
}

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
        let search_pattern = format!("%{query}%");
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

fn add_order_by_clause(
    qb: &mut QueryBuilder<sqlx::Postgres>,
    order_by: Option<&Vec<DatapointOrderBy>>,
    _search_query: Option<&str>,
) {
    let Some(order_by_vec) = order_by else {
        qb.push(" ORDER BY updated_at DESC, id DESC");
        return;
    };

    if order_by_vec.is_empty() {
        qb.push(" ORDER BY updated_at DESC, id DESC");
        return;
    }

    qb.push(" ORDER BY ");
    for (i, order_spec) in order_by_vec.iter().enumerate() {
        if i > 0 {
            qb.push(", ");
        }
        let column = match order_spec.term {
            DatapointOrderByTerm::Timestamp => "updated_at",
            DatapointOrderByTerm::SearchRelevance => {
                // For Postgres, we don't have the same term frequency calculation as ClickHouse.
                // Fall back to updated_at for now.
                // TODO(#5691): Implement proper text search ranking in Postgres
                "updated_at"
            }
        };
        let direction = match order_spec.direction {
            OrderDirection::Asc => "ASC",
            OrderDirection::Desc => "DESC",
        };
        qb.push(column);
        qb.push(" ");
        qb.push(direction);
    }
}

fn sort_datapoints(datapoints: &mut [StoredDatapoint], order_by: Option<&Vec<DatapointOrderBy>>) {
    // Default: sort by updated_at DESC, id DESC
    if let Some(order_by_vec) = order_by
        && !order_by_vec.is_empty()
    {
        // Use the first order_by term for sorting
        let first_order = &order_by_vec[0];
        let is_asc = matches!(first_order.direction, OrderDirection::Asc);
        match first_order.term {
            DatapointOrderByTerm::Timestamp | DatapointOrderByTerm::SearchRelevance => {
                datapoints.sort_by(|a, b| {
                    let a_updated = get_updated_at(a);
                    let b_updated = get_updated_at(b);
                    let cmp = a_updated.cmp(b_updated);
                    if is_asc { cmp } else { cmp.reverse() }
                });
            }
        }
    } else {
        // Default: DESC by updated_at
        datapoints.sort_by(|a, b| {
            let a_updated = get_updated_at(a);
            let b_updated = get_updated_at(b);
            b_updated.cmp(a_updated).then_with(|| b.id().cmp(&a.id()))
        });
    }
}

fn get_updated_at(dp: &StoredDatapoint) -> &str {
    match dp {
        StoredDatapoint::Chat(c) => &c.updated_at,
        StoredDatapoint::Json(j) => &j.updated_at,
    }
}

// =====================================================================
// Helper functions for deleting datapoints
// =====================================================================

async fn delete_chat_datapoints(
    pool: &PgPool,
    dataset_name: &str,
    datapoint_ids: Option<&[Uuid]>,
) -> Result<u64, Error> {
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

    let result = qb.build().execute(pool).await.map_err(|e| {
        Error::new(ErrorDetails::PostgresConnection {
            message: format!("Failed to delete chat datapoints: {e}"),
        })
    })?;

    Ok(result.rows_affected())
}

async fn delete_json_datapoints(
    pool: &PgPool,
    dataset_name: &str,
    datapoint_ids: Option<&[Uuid]>,
) -> Result<u64, Error> {
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

    let result = qb.build().execute(pool).await.map_err(|e| {
        Error::new(ErrorDetails::PostgresConnection {
            message: format!("Failed to delete json datapoints: {e}"),
        })
    })?;

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
    let mut qb = QueryBuilder::new(
        r"
        SELECT
            id, function_name, episode_id, input, output, tool_params,
            tags, is_custom, source_inference_id, name, snapshot_hash
        FROM tensorzero.chat_datapoints
        WHERE id = ANY(",
    );
    qb.push_bind(&source_ids);
    qb.push(") AND staled_at IS NULL");

    let rows = qb.build().fetch_all(pool).await.map_err(Error::from)?;

    if rows.is_empty() {
        return Ok(());
    }

    // Insert cloned datapoints
    let mut insert_qb = QueryBuilder::new(
        r"
        INSERT INTO tensorzero.chat_datapoints (
            id, dataset_name, function_name, episode_id, input, output, tool_params,
            tags, is_custom, source_inference_id, name, snapshot_hash
        )
        ",
    );

    insert_qb.push_values(&rows, |mut b, row| {
        let source_id: Uuid = row.get("id");
        let new_id = id_map.get(&source_id).copied().unwrap_or_else(Uuid::now_v7);

        b.push_bind(new_id)
            .push_bind(target_dataset_name)
            .push_bind(row.get::<String, _>("function_name"))
            .push_bind(row.get::<Option<Uuid>, _>("episode_id"))
            .push_bind(row.get::<serde_json::Value, _>("input"))
            .push_bind(row.get::<Option<serde_json::Value>, _>("output"))
            .push_bind(row.get::<Option<serde_json::Value>, _>("tool_params"))
            .push_bind(row.get::<serde_json::Value, _>("tags"))
            .push_bind(row.get::<bool, _>("is_custom"))
            .push_bind(row.get::<Option<Uuid>, _>("source_inference_id"))
            .push_bind(row.get::<Option<String>, _>("name"))
            .push_bind(row.get::<Option<Vec<u8>>, _>("snapshot_hash"));
    });

    insert_qb.build().execute(pool).await.map_err(|e| {
        Error::new(ErrorDetails::PostgresConnection {
            message: format!("Failed to clone chat datapoints: {e}"),
        })
    })?;

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
    let mut qb = QueryBuilder::new(
        r"
        SELECT
            id, function_name, episode_id, input, output, output_schema,
            tags, is_custom, source_inference_id, name, snapshot_hash
        FROM tensorzero.json_datapoints
        WHERE id = ANY(",
    );
    qb.push_bind(&source_ids);
    qb.push(") AND staled_at IS NULL");

    let rows = qb.build().fetch_all(pool).await.map_err(Error::from)?;

    if rows.is_empty() {
        return Ok(());
    }

    // Insert cloned datapoints
    let mut insert_qb = QueryBuilder::new(
        r"
        INSERT INTO tensorzero.json_datapoints (
            id, dataset_name, function_name, episode_id, input, output, output_schema,
            tags, is_custom, source_inference_id, name, snapshot_hash
        )
        ",
    );

    insert_qb.push_values(&rows, |mut b, row| {
        let source_id: Uuid = row.get("id");
        let new_id = id_map.get(&source_id).copied().unwrap_or_else(Uuid::now_v7);

        b.push_bind(new_id)
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
            .push_bind(row.get::<Option<Vec<u8>>, _>("snapshot_hash"));
    });

    insert_qb.build().execute(pool).await.map_err(|e| {
        Error::new(ErrorDetails::PostgresConnection {
            message: format!("Failed to clone json datapoints: {e}"),
        })
    })?;

    Ok(())
}

async fn verify_created_ids(pool: &PgPool, new_ids: &[Uuid]) -> Result<HashSet<Uuid>, Error> {
    if new_ids.is_empty() {
        return Ok(HashSet::new());
    }

    let mut qb = QueryBuilder::new(
        r"
        SELECT id FROM (
            SELECT id FROM tensorzero.chat_datapoints WHERE id = ANY(",
    );
    qb.push_bind(new_ids);
    qb.push(") AND staled_at IS NULL UNION ALL SELECT id FROM tensorzero.json_datapoints WHERE id = ANY(");
    qb.push_bind(new_ids);
    qb.push(") AND staled_at IS NULL) AS combined");

    let rows = qb.build().fetch_all(pool).await.map_err(Error::from)?;

    Ok(rows.into_iter().map(|row| row.get("id")).collect())
}
