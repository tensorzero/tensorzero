//! Inference queries for Postgres.
//!
//! This module implements read and write operations for inference tables in Postgres.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use num_bigint::BigUint;
use serde_json::Value;
use sqlx::QueryBuilder;
use std::collections::HashMap;
use uuid::Uuid;

use crate::config::MetricConfigType;
use crate::config::{Config, MetricConfigLevel};
use crate::db::clickhouse::query_builder::{InferenceFilter, OrderByTerm, OrderDirection};
use crate::db::inferences::{
    CountInferencesParams, DEFAULT_INFERENCE_QUERY_LIMIT, FunctionInfo, InferenceMetadata,
    InferenceOutputSource, InferenceQueries, ListInferenceMetadataParams, ListInferencesParams,
    PaginationParams,
};
use crate::endpoints::stored_inferences::v1::types::{
    BooleanMetricFilter, FloatComparisonOperator, FloatMetricFilter, TagComparisonOperator,
    TagFilter, TimeComparisonOperator, TimeFilter,
};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfigType;
use crate::inference::types::{
    ChatInferenceDatabaseInsert, FunctionType, JsonInferenceDatabaseInsert,
};
use crate::stored_inference::{
    StoredChatInferenceDatabase, StoredInferenceDatabase, StoredJsonInference,
};
use crate::tool::ToolCallConfigDatabaseInsert;

use super::PostgresConnectionInfo;

impl FloatComparisonOperator {
    fn to_postgres_operator(self) -> &'static str {
        match self {
            FloatComparisonOperator::LessThan => "<",
            FloatComparisonOperator::LessThanOrEqual => "<=",
            FloatComparisonOperator::Equal => "=",
            FloatComparisonOperator::GreaterThan => ">",
            FloatComparisonOperator::GreaterThanOrEqual => ">=",
            FloatComparisonOperator::NotEqual => "!=",
        }
    }
}

impl TimeComparisonOperator {
    fn to_postgres_operator(self) -> &'static str {
        match self {
            TimeComparisonOperator::LessThan => "<",
            TimeComparisonOperator::LessThanOrEqual => "<=",
            TimeComparisonOperator::Equal => "=",
            TimeComparisonOperator::GreaterThan => ">",
            TimeComparisonOperator::GreaterThanOrEqual => ">=",
            TimeComparisonOperator::NotEqual => "!=",
        }
    }
}

impl TagComparisonOperator {
    fn to_postgres_operator(self) -> &'static str {
        match self {
            TagComparisonOperator::Equal => "=",
            TagComparisonOperator::NotEqual => "!=",
        }
    }
}

#[async_trait]
impl InferenceQueries for PostgresConnectionInfo {
    // ===== Read methods =====

    async fn list_inferences(
        &self,
        config: &Config,
        params: &ListInferencesParams<'_>,
    ) -> Result<Vec<StoredInferenceDatabase>, Error> {
        params.validate_pagination()?;

        let pool = self.get_pool_result()?;

        // Determine which table(s) to query based on function_name
        let function_config_type = match params.function_name {
            Some(fn_name) => Some(config.get_function(fn_name)?.config_type()),
            None => None,
        };

        let limit = if params.limit == 0 {
            DEFAULT_INFERENCE_QUERY_LIMIT
        } else {
            params.limit
        };

        let offset = params.offset as i64;

        let mut results = Vec::new();

        // For single-table queries, push offset to the database
        // For multi-table queries, we need to fetch (limit + offset) rows and apply offset in memory
        let is_single_table = function_config_type.is_some();
        let db_offset = if is_single_table { offset } else { 0 };
        let db_limit = if is_single_table {
            limit as i64
        } else {
            // Fetch enough rows to handle offset in memory
            (limit + params.offset) as i64
        };

        // Query chat inferences if needed
        if function_config_type.is_none() || function_config_type == Some(FunctionConfigType::Chat)
        {
            let chat_results =
                query_chat_inferences(pool, config, params, db_limit, db_offset).await?;
            results.extend(chat_results.into_iter().map(StoredInferenceDatabase::Chat));
        }

        // Query json inferences if needed
        if function_config_type.is_none() || function_config_type == Some(FunctionConfigType::Json)
        {
            let json_results =
                query_json_inferences(pool, config, params, db_limit, db_offset).await?;
            results.extend(json_results.into_iter().map(StoredInferenceDatabase::Json));
        }

        // Sort by timestamp descending (most recent first) - default ordering
        // When querying both tables, we need to sort the combined results
        if !is_single_table {
            results.sort_by(|a, b| {
                let ts_a = match a {
                    StoredInferenceDatabase::Chat(c) => c.timestamp,
                    StoredInferenceDatabase::Json(j) => j.timestamp,
                };
                let ts_b = match b {
                    StoredInferenceDatabase::Chat(c) => c.timestamp,
                    StoredInferenceDatabase::Json(j) => j.timestamp,
                };
                ts_b.cmp(&ts_a) // Descending order
            });
        }

        // Reverse for "After" pagination
        if matches!(params.pagination, Some(PaginationParams::After { .. })) {
            results.reverse();
        }

        // Apply offset and limit for multi-table queries
        if !is_single_table {
            if offset > 0 {
                results = results.into_iter().skip(offset as usize).collect();
            }
            results.truncate(limit as usize);
        }

        Ok(results)
    }

    async fn list_inference_metadata(
        &self,
        params: &ListInferenceMetadataParams,
    ) -> Result<Vec<InferenceMetadata>, Error> {
        let pool = self.get_pool_result()?;

        let limit = if params.limit == 0 {
            DEFAULT_INFERENCE_QUERY_LIMIT as i64
        } else {
            params.limit as i64
        };

        // Query both tables and combine results
        let mut results = Vec::new();

        // Query chat inferences metadata
        let chat_results = query_chat_inference_metadata(pool, params, limit).await?;
        results.extend(chat_results);

        // Query json inferences metadata
        let json_results = query_json_inference_metadata(pool, params, limit).await?;
        results.extend(json_results);

        // Sort by id descending (UUIDv7 is time-ordered)
        results.sort_by(|a, b| b.id.cmp(&a.id));

        // Reverse for "After" pagination
        if matches!(params.pagination, Some(PaginationParams::After { .. })) {
            results.reverse();
        }

        // Apply limit
        results.truncate(limit as usize);

        Ok(results)
    }

    async fn count_inferences(
        &self,
        config: &Config,
        params: &CountInferencesParams<'_>,
    ) -> Result<u64, Error> {
        // Validate: filters and search_query are not supported yet
        if params.filters.is_some() {
            return Err(Error::new(ErrorDetails::PostgresQuery {
                message: "Filters are not yet supported for Postgres count queries".to_string(),
            }));
        }
        if params.search_query_experimental.is_some() {
            return Err(Error::new(ErrorDetails::PostgresQuery {
                message: "Search queries are not yet supported for Postgres count queries"
                    .to_string(),
            }));
        }

        let pool = self.get_pool_result()?;

        // Determine which table(s) to query based on function_name
        let function_config_type = match params.function_name {
            Some(fn_name) => Some(config.get_function(fn_name)?.config_type()),
            None => None,
        };

        let mut total: u64 = 0;

        // Count chat inferences if needed
        if function_config_type.is_none() || function_config_type == Some(FunctionConfigType::Chat)
        {
            total += count_chat_inferences(pool, params).await?;
        }

        // Count json inferences if needed
        if function_config_type.is_none() || function_config_type == Some(FunctionConfigType::Json)
        {
            total += count_json_inferences(pool, params).await?;
        }

        Ok(total)
    }

    async fn get_function_info(
        &self,
        target_id: &Uuid,
        level: MetricConfigLevel,
    ) -> Result<Option<FunctionInfo>, Error> {
        let pool = self.get_pool_result()?;

        match level {
            MetricConfigLevel::Inference => {
                // Try chat inferences first
                let chat_result = sqlx::query_as!(
                    FunctionInfoRow,
                    r#"
                    SELECT function_name, variant_name, episode_id
                    FROM tensorzero.chat_inferences
                    WHERE id = $1
                    LIMIT 1
                    "#,
                    target_id
                )
                .fetch_optional(pool)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::PostgresQuery {
                        message: format!("Failed to query chat_inferences: {e}"),
                    })
                })?;

                if let Some(row) = chat_result {
                    return Ok(Some(FunctionInfo {
                        function_name: row.function_name,
                        function_type: FunctionType::Chat,
                        variant_name: row.variant_name,
                        episode_id: row.episode_id,
                    }));
                }

                // Try json inferences
                let json_result = sqlx::query_as!(
                    FunctionInfoRow,
                    r#"
                    SELECT function_name, variant_name, episode_id
                    FROM tensorzero.json_inferences
                    WHERE id = $1
                    LIMIT 1
                    "#,
                    target_id
                )
                .fetch_optional(pool)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::PostgresQuery {
                        message: format!("Failed to query json_inferences: {e}"),
                    })
                })?;

                if let Some(row) = json_result {
                    return Ok(Some(FunctionInfo {
                        function_name: row.function_name,
                        function_type: FunctionType::Json,
                        variant_name: row.variant_name,
                        episode_id: row.episode_id,
                    }));
                }

                Ok(None)
            }
            MetricConfigLevel::Episode => {
                // Try chat inferences first (get most recent for this episode)
                let chat_result = sqlx::query_as!(
                    FunctionInfoRow,
                    r#"
                    SELECT function_name, variant_name, episode_id
                    FROM tensorzero.chat_inferences
                    WHERE episode_id = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                    "#,
                    target_id
                )
                .fetch_optional(pool)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::PostgresQuery {
                        message: format!("Failed to query chat_inferences by episode: {e}"),
                    })
                })?;

                if let Some(row) = chat_result {
                    return Ok(Some(FunctionInfo {
                        function_name: row.function_name,
                        function_type: FunctionType::Chat,
                        variant_name: row.variant_name,
                        episode_id: row.episode_id,
                    }));
                }

                // Try json inferences
                let json_result = sqlx::query_as!(
                    FunctionInfoRow,
                    r#"
                    SELECT function_name, variant_name, episode_id
                    FROM tensorzero.json_inferences
                    WHERE episode_id = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                    "#,
                    target_id
                )
                .fetch_optional(pool)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::PostgresQuery {
                        message: format!("Failed to query json_inferences by episode: {e}"),
                    })
                })?;

                if let Some(row) = json_result {
                    return Ok(Some(FunctionInfo {
                        function_name: row.function_name,
                        function_type: FunctionType::Json,
                        variant_name: row.variant_name,
                        episode_id: row.episode_id,
                    }));
                }

                Ok(None)
            }
        }
    }

    async fn get_chat_inference_tool_params(
        &self,
        function_name: &str,
        inference_id: Uuid,
    ) -> Result<Option<ToolCallConfigDatabaseInsert>, Error> {
        let pool = self.get_pool_result()?;

        let result = sqlx::query!(
            r#"
            SELECT
                tool_params,
                dynamic_tools,
                dynamic_provider_tools,
                allowed_tools,
                tool_choice,
                parallel_tool_calls
            FROM tensorzero.chat_inferences
            WHERE function_name = $1 AND id = $2
            LIMIT 1
            "#,
            function_name,
            inference_id
        )
        .fetch_optional(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to query chat_inference tool params: {e}"),
            })
        })?;

        match result {
            Some(row) => {
                // Deserialize the tool params from the stored JSON columns
                let tool_params = deserialize_tool_params_from_row(
                    row.tool_params,
                    row.dynamic_tools,
                    row.dynamic_provider_tools,
                    row.allowed_tools,
                    row.tool_choice,
                    row.parallel_tool_calls,
                )?;
                Ok(tool_params)
            }
            None => Ok(None),
        }
    }

    async fn get_json_inference_output_schema(
        &self,
        function_name: &str,
        inference_id: Uuid,
    ) -> Result<Option<Value>, Error> {
        let pool = self.get_pool_result()?;

        let result = sqlx::query!(
            r#"
            SELECT output_schema
            FROM tensorzero.json_inferences
            WHERE function_name = $1 AND id = $2
            LIMIT 1
            "#,
            function_name,
            inference_id
        )
        .fetch_optional(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to query json_inference output_schema: {e}"),
            })
        })?;

        Ok(result.map(|r| r.output_schema))
    }

    async fn get_inference_output(
        &self,
        function_info: &FunctionInfo,
        inference_id: Uuid,
    ) -> Result<Option<String>, Error> {
        let pool = self.get_pool_result()?;

        match function_info.function_type {
            FunctionType::Chat => {
                let result = sqlx::query!(
                    r#"
                    SELECT output
                    FROM tensorzero.chat_inferences
                    WHERE id = $1
                      AND episode_id = $2
                      AND function_name = $3
                      AND variant_name = $4
                    LIMIT 1
                    "#,
                    inference_id,
                    function_info.episode_id,
                    function_info.function_name,
                    function_info.variant_name
                )
                .fetch_optional(pool)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::PostgresQuery {
                        message: format!("Failed to query chat_inference output: {e}"),
                    })
                })?;

                // Convert JSONB to string
                Ok(result.map(|r| r.output.to_string()))
            }
            FunctionType::Json => {
                let result = sqlx::query!(
                    r#"
                    SELECT output
                    FROM tensorzero.json_inferences
                    WHERE id = $1
                      AND episode_id = $2
                      AND function_name = $3
                      AND variant_name = $4
                    LIMIT 1
                    "#,
                    inference_id,
                    function_info.episode_id,
                    function_info.function_name,
                    function_info.variant_name
                )
                .fetch_optional(pool)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::PostgresQuery {
                        message: format!("Failed to query json_inference output: {e}"),
                    })
                })?;

                // Convert JSONB to string
                Ok(result.map(|r| r.output.to_string()))
            }
        }
    }

    // ===== Write methods =====

    async fn insert_chat_inferences(
        &self,
        rows: &[ChatInferenceDatabaseInsert],
    ) -> Result<(), Error> {
        if rows.is_empty() {
            return Ok(());
        }

        let pool = self.get_pool_result()?;

        let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
            r"
            INSERT INTO tensorzero.chat_inferences (
                id, function_name, variant_name, episode_id, input, output,
                tool_params, inference_params, processing_time_ms, ttft_ms,
                tags, extra_body, dynamic_tools, dynamic_provider_tools,
                allowed_tools, tool_choice, parallel_tool_calls, snapshot_hash, created_at
            ) ",
        );

        query_builder.push_values(rows, |mut b, row| {
            let input_json = serde_json::to_value(&row.input).unwrap_or_default();
            let output_json = serde_json::to_value(&row.output).unwrap_or_default();
            let inference_params_json =
                serde_json::to_value(&row.inference_params).unwrap_or_default();
            let tags_json = serde_json::to_value(&row.tags).unwrap_or_default();
            let extra_body_json = serde_json::to_value(&row.extra_body).unwrap_or_default();

            let SerializedToolParams {
                tool_params_json,
                dynamic_tools,
                dynamic_provider_tools,
                allowed_tools,
                tool_choice,
                parallel_tool_calls,
            } = serialize_tool_params(row.tool_params.as_ref()).unwrap_or_default();

            let created_at = uuid_v7_to_timestamp(row.id);
            let snapshot_hash_bytes: Option<Vec<u8>> =
                row.snapshot_hash.as_ref().map(|h| h.as_bytes().to_vec());

            b.push_bind(row.id)
                .push_bind(&row.function_name)
                .push_bind(&row.variant_name)
                .push_bind(row.episode_id)
                .push_bind(input_json)
                .push_bind(output_json)
                .push_bind(tool_params_json)
                .push_bind(inference_params_json)
                .push_bind(row.processing_time_ms.map(|v| v as i32))
                .push_bind(row.ttft_ms.map(|v| v as i32))
                .push_bind(tags_json)
                .push_bind(extra_body_json)
                .push_bind(dynamic_tools)
                .push_bind(dynamic_provider_tools)
                .push_bind(allowed_tools)
                .push_bind(tool_choice)
                .push_bind(parallel_tool_calls)
                .push_bind(snapshot_hash_bytes)
                .push_bind(created_at);
        });

        query_builder.build().execute(pool).await.map_err(|e| {
            Error::new(ErrorDetails::PostgresConnection {
                message: format!("Failed to insert chat inferences: {e}"),
            })
        })?;

        Ok(())
    }

    async fn insert_json_inferences(
        &self,
        rows: &[JsonInferenceDatabaseInsert],
    ) -> Result<(), Error> {
        if rows.is_empty() {
            return Ok(());
        }

        let pool = self.get_pool_result()?;

        let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
            r"
            INSERT INTO tensorzero.json_inferences (
                id, function_name, variant_name, episode_id, input, output,
                output_schema, inference_params, processing_time_ms, ttft_ms,
                tags, extra_body, auxiliary_content, snapshot_hash, created_at
            ) ",
        );

        query_builder.push_values(rows, |mut b, row| {
            let input_json = serde_json::to_value(&row.input).unwrap_or_default();
            let output_json = serde_json::to_value(&row.output).unwrap_or_default();
            let output_schema_json = serde_json::to_value(&row.output_schema).unwrap_or_default();
            let inference_params_json =
                serde_json::to_value(&row.inference_params).unwrap_or_default();
            let tags_json = serde_json::to_value(&row.tags).unwrap_or_default();
            let extra_body_json = serde_json::to_value(&row.extra_body).unwrap_or_default();
            let auxiliary_content_json =
                serde_json::to_value(&row.auxiliary_content).unwrap_or_default();

            let created_at = uuid_v7_to_timestamp(row.id);
            let snapshot_hash_bytes: Option<Vec<u8>> =
                row.snapshot_hash.as_ref().map(|h| h.as_bytes().to_vec());

            b.push_bind(row.id)
                .push_bind(&row.function_name)
                .push_bind(&row.variant_name)
                .push_bind(row.episode_id)
                .push_bind(input_json)
                .push_bind(output_json)
                .push_bind(output_schema_json)
                .push_bind(inference_params_json)
                .push_bind(row.processing_time_ms.map(|v| v as i32))
                .push_bind(row.ttft_ms.map(|v| v as i32))
                .push_bind(tags_json)
                .push_bind(extra_body_json)
                .push_bind(auxiliary_content_json)
                .push_bind(snapshot_hash_bytes)
                .push_bind(created_at);
        });

        query_builder.build().execute(pool).await.map_err(|e| {
            Error::new(ErrorDetails::PostgresConnection {
                message: format!("Failed to insert json inferences: {e}"),
            })
        })?;

        Ok(())
    }
}

// ===== Helper types =====

/// Row type for function info queries.
struct FunctionInfoRow {
    function_name: String,
    variant_name: String,
    episode_id: Uuid,
}

/// Row type for chat inference queries with sqlx FromRow.
#[derive(sqlx::FromRow)]
#[expect(dead_code)]
struct ChatInferenceRow {
    id: Uuid,
    function_name: String,
    variant_name: String,
    episode_id: Uuid,
    timestamp: DateTime<Utc>,
    input: Value,
    output: Value,
    tool_params: Value,
    dynamic_tools: Value,
    dynamic_provider_tools: Value,
    allowed_tools: Option<Value>,
    tool_choice: Option<String>,
    parallel_tool_calls: Option<bool>,
    tags: Value,
    extra_body: Value,
    inference_params: Value,
    processing_time_ms: Option<i32>,
    ttft_ms: Option<i32>,
}

/// Row type for chat inference queries that may include dispreferred output (for demonstration source).
#[derive(sqlx::FromRow)]
struct ChatInferenceRowWithDispreferred {
    id: Uuid,
    function_name: String,
    variant_name: String,
    episode_id: Uuid,
    timestamp: DateTime<Utc>,
    input: Value,
    output: Value,
    /// The original output when using demonstration output source.
    /// This becomes a dispreferred output.
    dispreferred_output: Option<Value>,
    tool_params: Value,
    dynamic_tools: Value,
    dynamic_provider_tools: Value,
    allowed_tools: Option<Value>,
    tool_choice: Option<String>,
    parallel_tool_calls: Option<bool>,
    tags: Value,
    extra_body: Value,
    inference_params: Value,
    processing_time_ms: Option<i32>,
    ttft_ms: Option<i32>,
}

/// Row type for json inference queries with sqlx FromRow.
#[derive(sqlx::FromRow)]
#[expect(dead_code)]
struct JsonInferenceRow {
    id: Uuid,
    function_name: String,
    variant_name: String,
    episode_id: Uuid,
    timestamp: DateTime<Utc>,
    input: Value,
    output: Value,
    output_schema: Value,
    tags: Value,
    extra_body: Value,
    inference_params: Value,
    processing_time_ms: Option<i32>,
    ttft_ms: Option<i32>,
}

/// Row type for json inference queries that may include dispreferred output (for demonstration source).
#[derive(sqlx::FromRow)]
struct JsonInferenceRowWithDispreferred {
    id: Uuid,
    function_name: String,
    variant_name: String,
    episode_id: Uuid,
    timestamp: DateTime<Utc>,
    input: Value,
    output: Value,
    /// The original output when using demonstration output source.
    /// This becomes a dispreferred output.
    dispreferred_output: Option<Value>,
    output_schema: Value,
    tags: Value,
    extra_body: Value,
    inference_params: Value,
    processing_time_ms: Option<i32>,
    ttft_ms: Option<i32>,
}

/// Row type for inference metadata queries.
#[derive(sqlx::FromRow)]
struct InferenceMetadataRow {
    id: Uuid,
    function_name: String,
    variant_name: String,
    episode_id: Uuid,
    /// Snapshot hash stored as BYTEA in Postgres.
    /// Converted to decimal string representation when returning InferenceMetadata.
    snapshot_hash: Option<Vec<u8>>,
}

/// Context for building Postgres filter SQL.
/// Tracks state needed during recursive filter building.
struct PostgresFilterContext<'a> {
    config: &'a Config,
}

impl<'a> PostgresFilterContext<'a> {
    fn new(config: &'a Config) -> Self {
        Self { config }
    }

    /// Converts an InferenceFilter to Postgres SQL and pushes it to the query builder.
    /// Returns an error if a metric name is invalid.
    fn apply_filter(
        &self,
        query_builder: &mut QueryBuilder<sqlx::Postgres>,
        filter: &InferenceFilter,
    ) -> Result<(), Error> {
        match filter {
            InferenceFilter::FloatMetric(fm) => self.apply_float_metric_filter(query_builder, fm),
            InferenceFilter::BooleanMetric(bm) => {
                self.apply_boolean_metric_filter(query_builder, bm)
            }
            InferenceFilter::DemonstrationFeedback(df) => {
                Self::apply_demonstration_feedback_filter(query_builder, df);
                Ok(())
            }
            InferenceFilter::Tag(tag) => {
                Self::apply_tag_filter(query_builder, tag);
                Ok(())
            }
            InferenceFilter::Time(time) => {
                Self::apply_time_filter(query_builder, time);
                Ok(())
            }
            InferenceFilter::And { children } => self.apply_and_filter(query_builder, children),
            InferenceFilter::Or { children } => self.apply_or_filter(query_builder, children),
            InferenceFilter::Not { child } => self.apply_not_filter(query_builder, child),
        }
    }

    fn apply_float_metric_filter(
        &self,
        query_builder: &mut QueryBuilder<sqlx::Postgres>,
        fm: &FloatMetricFilter,
    ) -> Result<(), Error> {
        let metric_config = self.config.metrics.get(&fm.metric_name).ok_or_else(|| {
            Error::new(ErrorDetails::InvalidMetricName {
                metric_name: fm.metric_name.clone(),
            })
        })?;

        // Validate metric type
        if metric_config.r#type != MetricConfigType::Float {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!("Metric `{}` is not a float metric", fm.metric_name),
            }));
        }

        let join_column = match metric_config.level {
            MetricConfigLevel::Inference => "i.id",
            MetricConfigLevel::Episode => "i.episode_id",
        };
        let operator = fm.comparison_operator.to_postgres_operator();

        // Use EXISTS subquery to filter by metric value
        // We use argMax equivalent: ORDER BY created_at DESC LIMIT 1 to get latest feedback
        query_builder.push(
            " AND EXISTS (SELECT 1 FROM tensorzero.float_metric_feedback f WHERE f.target_id = ",
        );
        query_builder.push(join_column);
        query_builder.push(" AND f.metric_name = ");
        query_builder.push_bind(fm.metric_name.clone());
        query_builder.push(" AND f.value ");
        query_builder.push(operator);
        query_builder.push(" ");
        query_builder.push_bind(fm.value);
        query_builder.push(")");

        Ok(())
    }

    fn apply_boolean_metric_filter(
        &self,
        query_builder: &mut QueryBuilder<sqlx::Postgres>,
        bm: &BooleanMetricFilter,
    ) -> Result<(), Error> {
        let metric_config = self.config.metrics.get(&bm.metric_name).ok_or_else(|| {
            Error::new(ErrorDetails::InvalidMetricName {
                metric_name: bm.metric_name.clone(),
            })
        })?;

        // Validate metric type
        if metric_config.r#type != MetricConfigType::Boolean {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!("Metric `{}` is not a boolean metric", bm.metric_name),
            }));
        }

        let join_column = match metric_config.level {
            MetricConfigLevel::Inference => "i.id",
            MetricConfigLevel::Episode => "i.episode_id",
        };

        // Use EXISTS subquery to filter by metric value
        query_builder.push(
            " AND EXISTS (SELECT 1 FROM tensorzero.boolean_metric_feedback f WHERE f.target_id = ",
        );
        query_builder.push(join_column);
        query_builder.push(" AND f.metric_name = ");
        query_builder.push_bind(bm.metric_name.clone());
        query_builder.push(" AND f.value = ");
        query_builder.push_bind(bm.value);
        query_builder.push(")");

        Ok(())
    }

    fn apply_demonstration_feedback_filter(
        query_builder: &mut QueryBuilder<sqlx::Postgres>,
        df: &crate::endpoints::stored_inferences::v1::types::DemonstrationFeedbackFilter,
    ) {
        if df.has_demonstration {
            query_builder.push(
                " AND i.id IN (SELECT DISTINCT inference_id FROM tensorzero.demonstration_feedback)",
            );
        } else {
            query_builder.push(
                " AND i.id NOT IN (SELECT DISTINCT inference_id FROM tensorzero.demonstration_feedback)",
            );
        }
    }

    fn apply_tag_filter(query_builder: &mut QueryBuilder<sqlx::Postgres>, tag: &TagFilter) {
        let operator = tag.comparison_operator.to_postgres_operator();

        // For Postgres JSONB, we use the ->> operator to extract text and compare
        // We also check that the key exists to handle != correctly
        query_builder.push(" AND (i.tags ? ");
        query_builder.push_bind(tag.key.clone());
        query_builder.push(" AND i.tags->>");
        query_builder.push_bind(tag.key.clone());
        query_builder.push(" ");
        query_builder.push(operator);
        query_builder.push(" ");
        query_builder.push_bind(tag.value.clone());
        query_builder.push(")");
    }

    fn apply_time_filter(query_builder: &mut QueryBuilder<sqlx::Postgres>, time: &TimeFilter) {
        let operator = time.comparison_operator.to_postgres_operator();

        query_builder.push(" AND i.created_at ");
        query_builder.push(operator);
        query_builder.push(" ");
        query_builder.push_bind(time.time);
    }

    fn apply_and_filter(
        &self,
        query_builder: &mut QueryBuilder<sqlx::Postgres>,
        children: &[InferenceFilter],
    ) -> Result<(), Error> {
        // Empty AND is vacuously true - don't add any condition
        if children.is_empty() {
            return Ok(());
        }

        // For AND, we just apply all children filters sequentially
        // Each child adds its own " AND ..." clause
        for child in children {
            self.apply_filter(query_builder, child)?;
        }

        Ok(())
    }

    fn apply_or_filter(
        &self,
        query_builder: &mut QueryBuilder<sqlx::Postgres>,
        children: &[InferenceFilter],
    ) -> Result<(), Error> {
        // Empty OR is false - add a condition that's always false
        if children.is_empty() {
            query_builder.push(" AND FALSE");
            return Ok(());
        }

        // For OR, we need to group the conditions
        // We build each child's condition as a subexpression
        query_builder.push(" AND (");

        for (i, child) in children.iter().enumerate() {
            if i > 0 {
                query_builder.push(" OR ");
            }
            self.apply_or_child(query_builder, child)?;
        }

        query_builder.push(")");

        Ok(())
    }

    /// Applies a single child of an OR filter.
    /// This needs special handling because we need to wrap the condition properly.
    fn apply_or_child(
        &self,
        query_builder: &mut QueryBuilder<sqlx::Postgres>,
        filter: &InferenceFilter,
    ) -> Result<(), Error> {
        match filter {
            InferenceFilter::FloatMetric(fm) => {
                let metric_config = self.config.metrics.get(&fm.metric_name).ok_or_else(|| {
                    Error::new(ErrorDetails::InvalidMetricName {
                        metric_name: fm.metric_name.clone(),
                    })
                })?;
                if metric_config.r#type != MetricConfigType::Float {
                    return Err(Error::new(ErrorDetails::InvalidRequest {
                        message: format!("Metric `{}` is not a float metric", fm.metric_name),
                    }));
                }
                let join_column = match metric_config.level {
                    MetricConfigLevel::Inference => "i.id",
                    MetricConfigLevel::Episode => "i.episode_id",
                };
                let operator = fm.comparison_operator.to_postgres_operator();

                query_builder.push(
                    "EXISTS (SELECT 1 FROM tensorzero.float_metric_feedback f WHERE f.target_id = ",
                );
                query_builder.push(join_column);
                query_builder.push(" AND f.metric_name = ");
                query_builder.push_bind(fm.metric_name.clone());
                query_builder.push(" AND f.value ");
                query_builder.push(operator);
                query_builder.push(" ");
                query_builder.push_bind(fm.value);
                query_builder.push(")");
            }
            InferenceFilter::BooleanMetric(bm) => {
                let metric_config = self.config.metrics.get(&bm.metric_name).ok_or_else(|| {
                    Error::new(ErrorDetails::InvalidMetricName {
                        metric_name: bm.metric_name.clone(),
                    })
                })?;
                if metric_config.r#type != MetricConfigType::Boolean {
                    return Err(Error::new(ErrorDetails::InvalidRequest {
                        message: format!("Metric `{}` is not a boolean metric", bm.metric_name),
                    }));
                }
                let join_column = match metric_config.level {
                    MetricConfigLevel::Inference => "i.id",
                    MetricConfigLevel::Episode => "i.episode_id",
                };

                query_builder.push("EXISTS (SELECT 1 FROM tensorzero.boolean_metric_feedback f WHERE f.target_id = ");
                query_builder.push(join_column);
                query_builder.push(" AND f.metric_name = ");
                query_builder.push_bind(bm.metric_name.clone());
                query_builder.push(" AND f.value = ");
                query_builder.push_bind(bm.value);
                query_builder.push(")");
            }
            InferenceFilter::DemonstrationFeedback(df) => {
                if df.has_demonstration {
                    query_builder.push(
                        "i.id IN (SELECT DISTINCT inference_id FROM tensorzero.demonstration_feedback)",
                    );
                } else {
                    query_builder.push(
                        "i.id NOT IN (SELECT DISTINCT inference_id FROM tensorzero.demonstration_feedback)",
                    );
                }
            }
            InferenceFilter::Tag(tag) => {
                let operator = tag.comparison_operator.to_postgres_operator();
                query_builder.push("(i.tags ? ");
                query_builder.push_bind(tag.key.clone());
                query_builder.push(" AND i.tags->>");
                query_builder.push_bind(tag.key.clone());
                query_builder.push(" ");
                query_builder.push(operator);
                query_builder.push(" ");
                query_builder.push_bind(tag.value.clone());
                query_builder.push(")");
            }
            InferenceFilter::Time(time) => {
                let operator = time.comparison_operator.to_postgres_operator();
                query_builder.push("i.created_at ");
                query_builder.push(operator);
                query_builder.push(" ");
                query_builder.push_bind(time.time);
            }
            InferenceFilter::And { children } => {
                if children.is_empty() {
                    query_builder.push("TRUE");
                } else {
                    query_builder.push("(");
                    for (i, child) in children.iter().enumerate() {
                        if i > 0 {
                            query_builder.push(" AND ");
                        }
                        self.apply_or_child(query_builder, child)?;
                    }
                    query_builder.push(")");
                }
            }
            InferenceFilter::Or { children } => {
                if children.is_empty() {
                    query_builder.push("FALSE");
                } else {
                    query_builder.push("(");
                    for (i, child) in children.iter().enumerate() {
                        if i > 0 {
                            query_builder.push(" OR ");
                        }
                        self.apply_or_child(query_builder, child)?;
                    }
                    query_builder.push(")");
                }
            }
            InferenceFilter::Not { child } => {
                query_builder.push("NOT (");
                self.apply_or_child(query_builder, child)?;
                query_builder.push(")");
            }
        }
        Ok(())
    }

    fn apply_not_filter(
        &self,
        query_builder: &mut QueryBuilder<sqlx::Postgres>,
        child: &InferenceFilter,
    ) -> Result<(), Error> {
        query_builder.push(" AND NOT (");
        self.apply_or_child(query_builder, child)?;
        query_builder.push(")");
        Ok(())
    }
}

/// Applies the inference filter to a query builder.
/// The query builder should already have a WHERE clause started (e.g., WHERE 1=1).
fn apply_inference_filter(
    query_builder: &mut QueryBuilder<sqlx::Postgres>,
    filter: Option<&InferenceFilter>,
    config: &Config,
) -> Result<(), Error> {
    if let Some(f) = filter {
        let ctx = PostgresFilterContext::new(config);
        ctx.apply_filter(query_builder, f)?;
    }
    Ok(())
}

/// Tracks metric JOINs needed for ORDER BY clauses.
struct MetricJoinRegistry {
    /// JOIN clauses to add to the query.
    joins: Vec<String>,
    /// Counter for generating unique join aliases.
    alias_counter: usize,
}

impl MetricJoinRegistry {
    fn new() -> Self {
        Self {
            joins: Vec::new(),
            alias_counter: 0,
        }
    }

    /// Registers a metric join and returns the alias for the joined value.
    /// Uses DISTINCT ON to get the latest feedback value per target.
    fn register_metric_join(
        &mut self,
        metric_name: &str,
        metric_type: MetricConfigType,
        level: MetricConfigLevel,
    ) -> String {
        let alias = format!("metric_{}", self.alias_counter);
        self.alias_counter += 1;

        let table_name = match metric_type {
            MetricConfigType::Float => "tensorzero.float_metric_feedback",
            MetricConfigType::Boolean => "tensorzero.boolean_metric_feedback",
        };

        let inference_column = level.inference_column_name();

        // Use a subquery with DISTINCT ON to get the latest feedback value per target
        let join_clause = format!(
            r"
LEFT JOIN (
    SELECT DISTINCT ON (target_id)
        target_id,
        value
    FROM {table_name}
    WHERE metric_name = '{metric_name}'
    ORDER BY target_id, created_at DESC
) AS {alias} ON i.{inference_column} = {alias}.target_id"
        );

        self.joins.push(join_clause);
        alias
    }

    /// Returns the JOIN clauses as a single string.
    fn get_joins_sql(&self) -> String {
        self.joins.join("")
    }
}

/// Result of building ORDER BY clause, including any required JOINs.
struct OrderByResult {
    /// The ORDER BY clause (e.g., " ORDER BY i.created_at DESC, i.id DESC").
    order_by_clause: String,
    /// JOIN clauses needed for metric ordering.
    metric_joins: String,
}

/// Builds the ORDER BY clause based on params.
/// For "After" pagination, we invert the direction since we'll reverse results in memory.
/// Returns both the ORDER BY clause and any required metric JOINs.
fn build_order_by_clause(
    params: &ListInferencesParams<'_>,
    config: &Config,
) -> Result<OrderByResult, Error> {
    let should_invert_directions =
        matches!(params.pagination, Some(PaginationParams::After { .. }));

    // Build id tie-breaker direction
    let id_direction = if let Some(ref pagination) = params.pagination {
        match pagination {
            PaginationParams::After { .. } => OrderDirection::Asc,
            PaginationParams::Before { .. } => OrderDirection::Desc,
        }
    } else {
        OrderDirection::Desc // Default: most recent first
    };

    let mut order_clauses = Vec::new();
    let mut join_registry = MetricJoinRegistry::new();

    // Add user-specified ordering if present
    if let Some(order_by) = params.order_by {
        for o in order_by {
            let column = match &o.term {
                OrderByTerm::Timestamp => "i.created_at".to_string(),
                OrderByTerm::Metric { name } => {
                    // Look up metric config to determine table and join column
                    let metric_config = config.metrics.get(name).ok_or_else(|| {
                        Error::new(ErrorDetails::InvalidMetricName {
                            metric_name: name.clone(),
                        })
                    })?;

                    // All metric types (Float, Boolean) are orderable

                    // Register the join and get the alias
                    let alias = join_registry.register_metric_join(
                        name,
                        metric_config.r#type,
                        metric_config.level.clone(),
                    );
                    format!("{alias}.value")
                }
                OrderByTerm::SearchRelevance => {
                    if params.search_query_experimental.is_none() {
                        return Err(Error::new(ErrorDetails::InvalidRequest {
                            message: "ORDER BY search_relevance requires search_query_experimental to be provided".to_string(),
                        }));
                    }
                    // For Postgres, we don't have a term frequency column
                    // This is a known limitation compared to ClickHouse
                    return Err(Error::new(ErrorDetails::PostgresQuery {
                        message: "ORDER BY search_relevance is not yet supported for Postgres inference queries".to_string(),
                    }));
                }
            };

            let effective_direction = if should_invert_directions {
                o.direction.inverted()
            } else {
                o.direction
            };

            // Add NULLS LAST to handle NULL metric values gracefully
            order_clauses.push(format!(
                "{} {} NULLS LAST",
                column,
                effective_direction.to_clickhouse_direction()
            ));
        }
    }

    // Always add id as tie-breaker for deterministic ordering
    order_clauses.push(format!("i.id {}", id_direction.to_clickhouse_direction()));

    Ok(OrderByResult {
        order_by_clause: format!(" ORDER BY {}", order_clauses.join(", ")),
        metric_joins: join_registry.get_joins_sql(),
    })
}

// ===== Helper functions for chat inference queries =====

async fn query_chat_inferences(
    pool: &sqlx::PgPool,
    config: &Config,
    params: &ListInferencesParams<'_>,
    limit: i64,
    offset: i64,
) -> Result<Vec<StoredChatInferenceDatabase>, Error> {
    // Build ORDER BY clause first to get any required metric JOINs
    let order_by_result = build_order_by_clause(params, config)?;

    // Build the SELECT clause based on output_source
    let output_select = match params.output_source {
        InferenceOutputSource::None | InferenceOutputSource::Inference => {
            "i.output, NULL::jsonb as dispreferred_output"
        }
        InferenceOutputSource::Demonstration => {
            "demo_f.value AS output, i.output as dispreferred_output"
        }
    };

    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(format!(
        r"
        SELECT
            i.id,
            i.function_name,
            i.variant_name,
            i.episode_id,
            i.created_at as timestamp,
            i.input,
            {output_select},
            i.tool_params,
            i.dynamic_tools,
            i.dynamic_provider_tools,
            i.allowed_tools,
            i.tool_choice,
            i.parallel_tool_calls,
            i.tags,
            i.extra_body,
            i.inference_params,
            i.processing_time_ms,
            i.ttft_ms
        FROM tensorzero.chat_inferences i
        "
    ));

    // Add JOIN for demonstration output source
    if params.output_source == InferenceOutputSource::Demonstration {
        query_builder.push(
            r"
            JOIN (
                SELECT DISTINCT ON (inference_id)
                    inference_id,
                    value
                FROM tensorzero.demonstration_feedback
                ORDER BY inference_id, created_at DESC
            ) AS demo_f ON i.id = demo_f.inference_id
            ",
        );
    }

    // Add metric JOINs for ORDER BY
    if !order_by_result.metric_joins.is_empty() {
        query_builder.push(&order_by_result.metric_joins);
    }

    query_builder.push(" WHERE 1=1");

    // Add filters
    if let Some(function_name) = params.function_name {
        query_builder.push(" AND i.function_name = ");
        query_builder.push_bind(function_name);
    }

    if let Some(variant_name) = params.variant_name {
        query_builder.push(" AND i.variant_name = ");
        query_builder.push_bind(variant_name);
    }

    if let Some(episode_id) = params.episode_id {
        query_builder.push(" AND i.episode_id = ");
        query_builder.push_bind(*episode_id);
    }

    if let Some(ids) = params.ids {
        query_builder.push(" AND i.id = ANY(");
        query_builder.push_bind(ids);
        query_builder.push(")");
    }

    // Apply inference filter (e.g., DemonstrationFeedback, metric filters, etc.)
    apply_inference_filter(&mut query_builder, params.filters, config)?;

    // Apply search query filter
    if let Some(search_query) = params.search_query_experimental {
        let search_pattern = format!("%{search_query}%");
        query_builder.push(" AND (i.input::text ILIKE ");
        query_builder.push_bind(search_pattern.clone());
        query_builder.push(" OR i.output::text ILIKE ");
        query_builder.push_bind(search_pattern);
        query_builder.push(")");
    }

    // Handle pagination cursor
    match &params.pagination {
        Some(PaginationParams::Before { id }) => {
            query_builder.push(" AND i.id < ");
            query_builder.push_bind(*id);
        }
        Some(PaginationParams::After { id }) => {
            query_builder.push(" AND i.id > ");
            query_builder.push_bind(*id);
        }
        None => {}
    }

    // Add ORDER BY clause
    query_builder.push(&order_by_result.order_by_clause);

    // Add LIMIT
    query_builder.push(" LIMIT ");
    query_builder.push_bind(limit);

    // Add OFFSET
    if offset > 0 {
        query_builder.push(" OFFSET ");
        query_builder.push_bind(offset);
    }

    let rows: Vec<ChatInferenceRowWithDispreferred> = query_builder
        .build_query_as()
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to query chat_inferences: {e}"),
            })
        })?;

    // Convert rows to StoredChatInferenceDatabase
    rows.into_iter()
        .map(convert_chat_row_with_dispreferred_to_stored)
        .collect()
}

fn convert_chat_row_with_dispreferred_to_stored(
    row: ChatInferenceRowWithDispreferred,
) -> Result<StoredChatInferenceDatabase, Error> {
    use crate::endpoints::inference::InferenceParams;
    use crate::inference::types::ContentBlockChatOutput;
    use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
    use crate::inference::types::stored_input::StoredInput;

    let input: StoredInput = serde_json::from_value(row.input).map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to deserialize chat inference input: {e}"),
        })
    })?;

    let output: Vec<ContentBlockChatOutput> = serde_json::from_value(row.output).map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to deserialize chat inference output: {e}"),
        })
    })?;

    let tags: HashMap<String, String> = serde_json::from_value(row.tags).map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to deserialize chat inference tags: {e}"),
        })
    })?;

    let extra_body: UnfilteredInferenceExtraBody =
        serde_json::from_value(row.extra_body).map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to deserialize chat inference extra_body: {e}"),
            })
        })?;

    let inference_params: InferenceParams =
        serde_json::from_value(row.inference_params).map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to deserialize chat inference inference_params: {e}"),
            })
        })?;

    // Reconstruct tool_params from the separate columns
    let tool_params = deserialize_tool_params_from_row(
        row.tool_params,
        row.dynamic_tools,
        row.dynamic_provider_tools,
        row.allowed_tools,
        row.tool_choice,
        row.parallel_tool_calls,
    )?
    .unwrap_or_default();

    // Build dispreferred_outputs from the dispreferred_output field
    let dispreferred_outputs = if let Some(dispreferred) = row.dispreferred_output {
        let dispreferred_output: Vec<ContentBlockChatOutput> = serde_json::from_value(dispreferred)
            .map_err(|e| {
                Error::new(ErrorDetails::PostgresQuery {
                    message: format!("Failed to deserialize dispreferred output: {e}"),
                })
            })?;
        vec![dispreferred_output]
    } else {
        vec![]
    };

    Ok(StoredChatInferenceDatabase {
        function_name: row.function_name,
        variant_name: row.variant_name,
        input,
        output,
        dispreferred_outputs,
        timestamp: row.timestamp,
        episode_id: row.episode_id,
        inference_id: row.id,
        tool_params,
        tags,
        extra_body,
        inference_params,
        processing_time_ms: row.processing_time_ms.map(|v| v as u64),
        ttft_ms: row.ttft_ms.map(|v| v as u64),
    })
}

async fn query_json_inferences(
    pool: &sqlx::PgPool,
    config: &Config,
    params: &ListInferencesParams<'_>,
    limit: i64,
    offset: i64,
) -> Result<Vec<StoredJsonInference>, Error> {
    // Build ORDER BY clause first to get any required metric JOINs
    let order_by_result = build_order_by_clause(params, config)?;

    // Build the SELECT clause based on output_source
    let output_select = match params.output_source {
        InferenceOutputSource::None | InferenceOutputSource::Inference => {
            "i.output, NULL::jsonb as dispreferred_output"
        }
        InferenceOutputSource::Demonstration => {
            "demo_f.value AS output, i.output as dispreferred_output"
        }
    };

    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(format!(
        r"
        SELECT
            i.id,
            i.function_name,
            i.variant_name,
            i.episode_id,
            i.created_at as timestamp,
            i.input,
            {output_select},
            i.output_schema,
            i.tags,
            i.extra_body,
            i.inference_params,
            i.processing_time_ms,
            i.ttft_ms
        FROM tensorzero.json_inferences i
        "
    ));

    // Add JOIN for demonstration output source
    if params.output_source == InferenceOutputSource::Demonstration {
        query_builder.push(
            r"
            JOIN (
                SELECT DISTINCT ON (inference_id)
                    inference_id,
                    value
                FROM tensorzero.demonstration_feedback
                ORDER BY inference_id, created_at DESC
            ) AS demo_f ON i.id = demo_f.inference_id
            ",
        );
    }

    // Add metric JOINs for ORDER BY
    if !order_by_result.metric_joins.is_empty() {
        query_builder.push(&order_by_result.metric_joins);
    }

    query_builder.push(" WHERE 1=1");

    // Add filters
    if let Some(function_name) = params.function_name {
        query_builder.push(" AND i.function_name = ");
        query_builder.push_bind(function_name);
    }

    if let Some(variant_name) = params.variant_name {
        query_builder.push(" AND i.variant_name = ");
        query_builder.push_bind(variant_name);
    }

    if let Some(episode_id) = params.episode_id {
        query_builder.push(" AND i.episode_id = ");
        query_builder.push_bind(*episode_id);
    }

    if let Some(ids) = params.ids {
        query_builder.push(" AND i.id = ANY(");
        query_builder.push_bind(ids);
        query_builder.push(")");
    }

    // Apply inference filter (e.g., DemonstrationFeedback, metric filters, etc.)
    apply_inference_filter(&mut query_builder, params.filters, config)?;

    // Apply search query filter
    if let Some(search_query) = params.search_query_experimental {
        let search_pattern = format!("%{search_query}%");
        query_builder.push(" AND (i.input::text ILIKE ");
        query_builder.push_bind(search_pattern.clone());
        query_builder.push(" OR i.output::text ILIKE ");
        query_builder.push_bind(search_pattern);
        query_builder.push(")");
    }

    // Handle pagination cursor
    match &params.pagination {
        Some(PaginationParams::Before { id }) => {
            query_builder.push(" AND i.id < ");
            query_builder.push_bind(*id);
        }
        Some(PaginationParams::After { id }) => {
            query_builder.push(" AND i.id > ");
            query_builder.push_bind(*id);
        }
        None => {}
    }

    // Add ORDER BY clause
    query_builder.push(&order_by_result.order_by_clause);

    // Add LIMIT
    query_builder.push(" LIMIT ");
    query_builder.push_bind(limit);

    // Add OFFSET
    if offset > 0 {
        query_builder.push(" OFFSET ");
        query_builder.push_bind(offset);
    }

    let rows: Vec<JsonInferenceRowWithDispreferred> = query_builder
        .build_query_as()
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to query json_inferences: {e}"),
            })
        })?;

    // Convert rows to StoredJsonInference
    rows.into_iter()
        .map(convert_json_row_with_dispreferred_to_stored)
        .collect()
}

fn convert_json_row_with_dispreferred_to_stored(
    row: JsonInferenceRowWithDispreferred,
) -> Result<StoredJsonInference, Error> {
    use crate::endpoints::inference::InferenceParams;
    use crate::inference::types::JsonInferenceOutput;
    use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
    use crate::inference::types::stored_input::StoredInput;

    let input: StoredInput = serde_json::from_value(row.input).map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to deserialize json inference input: {e}"),
        })
    })?;

    let output: JsonInferenceOutput = serde_json::from_value(row.output).map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to deserialize json inference output: {e}"),
        })
    })?;

    let tags: HashMap<String, String> = serde_json::from_value(row.tags).map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to deserialize json inference tags: {e}"),
        })
    })?;

    let extra_body: UnfilteredInferenceExtraBody =
        serde_json::from_value(row.extra_body).map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to deserialize json inference extra_body: {e}"),
            })
        })?;

    let inference_params: InferenceParams =
        serde_json::from_value(row.inference_params).map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to deserialize json inference inference_params: {e}"),
            })
        })?;

    // Build dispreferred_outputs from the dispreferred_output field
    let dispreferred_outputs = if let Some(dispreferred) = row.dispreferred_output {
        let dispreferred_output: JsonInferenceOutput = serde_json::from_value(dispreferred)
            .map_err(|e| {
                Error::new(ErrorDetails::PostgresQuery {
                    message: format!("Failed to deserialize dispreferred output: {e}"),
                })
            })?;
        vec![dispreferred_output]
    } else {
        vec![]
    };

    Ok(StoredJsonInference {
        function_name: row.function_name,
        variant_name: row.variant_name,
        input,
        output,
        dispreferred_outputs,
        timestamp: row.timestamp,
        episode_id: row.episode_id,
        inference_id: row.id,
        output_schema: row.output_schema,
        tags,
        extra_body,
        inference_params,
        processing_time_ms: row.processing_time_ms.map(|v| v as u64),
        ttft_ms: row.ttft_ms.map(|v| v as u64),
    })
}

// ===== Helper functions for inference metadata queries =====

async fn query_chat_inference_metadata(
    pool: &sqlx::PgPool,
    params: &ListInferenceMetadataParams,
    limit: i64,
) -> Result<Vec<InferenceMetadata>, Error> {
    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
        r"
        SELECT id, function_name, variant_name, episode_id, snapshot_hash
        FROM tensorzero.chat_inferences
        WHERE 1=1
        ",
    );

    if let Some(ref function_name) = params.function_name {
        query_builder.push(" AND function_name = ");
        query_builder.push_bind(function_name);
    }

    if let Some(ref variant_name) = params.variant_name {
        query_builder.push(" AND variant_name = ");
        query_builder.push_bind(variant_name);
    }

    if let Some(episode_id) = params.episode_id {
        query_builder.push(" AND episode_id = ");
        query_builder.push_bind(episode_id);
    }

    // Handle pagination
    let order_direction = match &params.pagination {
        Some(PaginationParams::Before { id }) => {
            query_builder.push(" AND id < ");
            query_builder.push_bind(*id);
            "DESC"
        }
        Some(PaginationParams::After { id }) => {
            query_builder.push(" AND id > ");
            query_builder.push_bind(*id);
            "ASC"
        }
        None => "DESC",
    };

    query_builder.push(" ORDER BY id ");
    query_builder.push(order_direction);
    query_builder.push(" LIMIT ");
    query_builder.push_bind(limit);

    let rows: Vec<InferenceMetadataRow> = query_builder
        .build_query_as()
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to query chat_inferences metadata: {e}"),
            })
        })?;

    Ok(rows
        .into_iter()
        .map(|row| InferenceMetadata {
            id: row.id,
            function_name: row.function_name,
            variant_name: row.variant_name,
            episode_id: row.episode_id,
            function_type: FunctionType::Chat,
            // Convert BYTEA back to decimal string representation
            snapshot_hash: row
                .snapshot_hash
                .map(|bytes| BigUint::from_bytes_be(&bytes).to_string()),
        })
        .collect())
}

async fn query_json_inference_metadata(
    pool: &sqlx::PgPool,
    params: &ListInferenceMetadataParams,
    limit: i64,
) -> Result<Vec<InferenceMetadata>, Error> {
    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
        r"
        SELECT id, function_name, variant_name, episode_id, snapshot_hash
        FROM tensorzero.json_inferences
        WHERE 1=1
        ",
    );

    if let Some(ref function_name) = params.function_name {
        query_builder.push(" AND function_name = ");
        query_builder.push_bind(function_name);
    }

    if let Some(ref variant_name) = params.variant_name {
        query_builder.push(" AND variant_name = ");
        query_builder.push_bind(variant_name);
    }

    if let Some(episode_id) = params.episode_id {
        query_builder.push(" AND episode_id = ");
        query_builder.push_bind(episode_id);
    }

    // Handle pagination
    let order_direction = match &params.pagination {
        Some(PaginationParams::Before { id }) => {
            query_builder.push(" AND id < ");
            query_builder.push_bind(*id);
            "DESC"
        }
        Some(PaginationParams::After { id }) => {
            query_builder.push(" AND id > ");
            query_builder.push_bind(*id);
            "ASC"
        }
        None => "DESC",
    };

    query_builder.push(" ORDER BY id ");
    query_builder.push(order_direction);
    query_builder.push(" LIMIT ");
    query_builder.push_bind(limit);

    let rows: Vec<InferenceMetadataRow> = query_builder
        .build_query_as()
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to query json_inferences metadata: {e}"),
            })
        })?;

    Ok(rows
        .into_iter()
        .map(|row| InferenceMetadata {
            id: row.id,
            function_name: row.function_name,
            variant_name: row.variant_name,
            episode_id: row.episode_id,
            function_type: FunctionType::Json,
            // Convert BYTEA back to decimal string representation
            snapshot_hash: row
                .snapshot_hash
                .map(|bytes| BigUint::from_bytes_be(&bytes).to_string()),
        })
        .collect())
}

// ===== Helper functions for count queries =====

async fn count_chat_inferences(
    pool: &sqlx::PgPool,
    params: &CountInferencesParams<'_>,
) -> Result<u64, Error> {
    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
        r"
        SELECT COUNT(*)::BIGINT
        FROM tensorzero.chat_inferences
        WHERE 1=1
        ",
    );

    if let Some(function_name) = params.function_name {
        query_builder.push(" AND function_name = ");
        query_builder.push_bind(function_name);
    }

    if let Some(variant_name) = params.variant_name {
        query_builder.push(" AND variant_name = ");
        query_builder.push_bind(variant_name);
    }

    if let Some(episode_id) = params.episode_id {
        query_builder.push(" AND episode_id = ");
        query_builder.push_bind(*episode_id);
    }

    let query = query_builder.build_query_scalar::<i64>();
    let count: i64 = query.fetch_one(pool).await.map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to count chat_inferences: {e}"),
        })
    })?;

    Ok(count as u64)
}

async fn count_json_inferences(
    pool: &sqlx::PgPool,
    params: &CountInferencesParams<'_>,
) -> Result<u64, Error> {
    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
        r"
        SELECT COUNT(*)::BIGINT
        FROM tensorzero.json_inferences
        WHERE 1=1
        ",
    );

    if let Some(function_name) = params.function_name {
        query_builder.push(" AND function_name = ");
        query_builder.push_bind(function_name);
    }

    if let Some(variant_name) = params.variant_name {
        query_builder.push(" AND variant_name = ");
        query_builder.push_bind(variant_name);
    }

    if let Some(episode_id) = params.episode_id {
        query_builder.push(" AND episode_id = ");
        query_builder.push_bind(*episode_id);
    }

    let query = query_builder.build_query_scalar::<i64>();
    let count: i64 = query.fetch_one(pool).await.map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to count json_inferences: {e}"),
        })
    })?;

    Ok(count as u64)
}

// ===== Helper function to deserialize tool params =====

fn deserialize_tool_params_from_row(
    _tool_params: Value,
    dynamic_tools: Value,
    dynamic_provider_tools: Value,
    allowed_tools: Option<Value>,
    tool_choice: Option<String>,
    parallel_tool_calls: Option<bool>,
) -> Result<Option<ToolCallConfigDatabaseInsert>, Error> {
    // Check if we have any non-default values
    let has_dynamic_tools = dynamic_tools.as_array().is_some_and(|arr| !arr.is_empty());
    let has_provider_tools = dynamic_provider_tools
        .as_array()
        .is_some_and(|arr| !arr.is_empty());
    let has_allowed_tools = allowed_tools.is_some();
    let has_tool_choice = tool_choice.is_some();
    let has_parallel = parallel_tool_calls.is_some();

    // If no tool params are set, return default
    if !has_dynamic_tools
        && !has_provider_tools
        && !has_allowed_tools
        && !has_tool_choice
        && !has_parallel
    {
        return Ok(None);
    }

    // Build a JSON object that can be deserialized by the existing deserializer
    let mut obj = serde_json::Map::new();
    obj.insert("dynamic_tools".to_string(), dynamic_tools);
    obj.insert("dynamic_provider_tools".to_string(), dynamic_provider_tools);
    if let Some(allowed) = allowed_tools {
        obj.insert("allowed_tools".to_string(), allowed);
    }
    if let Some(choice) = tool_choice {
        obj.insert("tool_choice".to_string(), Value::String(choice));
    }
    if let Some(parallel) = parallel_tool_calls {
        obj.insert("parallel_tool_calls".to_string(), Value::Bool(parallel));
    }

    let result: ToolCallConfigDatabaseInsert =
        serde_json::from_value(Value::Object(obj)).map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to deserialize tool_params: {e}"),
            })
        })?;

    Ok(Some(result))
}

/// Unix epoch constant for fallback when timestamp extraction fails.
const UNIX_EPOCH: DateTime<Utc> = DateTime::from_timestamp(0, 0).unwrap();

/// Extract timestamp from UUIDv7.
/// UUIDv7 stores milliseconds since Unix epoch in the first 48 bits.
fn uuid_v7_to_timestamp(uuid: Uuid) -> DateTime<Utc> {
    let bytes = uuid.as_bytes();
    // First 6 bytes (48 bits) contain the timestamp in milliseconds
    let timestamp_ms = ((bytes[0] as u64) << 40)
        | ((bytes[1] as u64) << 32)
        | ((bytes[2] as u64) << 24)
        | ((bytes[3] as u64) << 16)
        | ((bytes[4] as u64) << 8)
        | (bytes[5] as u64);

    DateTime::from_timestamp_millis(timestamp_ms as i64).unwrap_or(UNIX_EPOCH)
}

/// Serialized tool params for Postgres columns.
#[derive(Default)]
struct SerializedToolParams {
    tool_params_json: Value,
    dynamic_tools: Value,
    dynamic_provider_tools: Value,
    allowed_tools: Option<Value>,
    tool_choice: Option<String>,
    parallel_tool_calls: Option<bool>,
}

/// Serialize tool params into separate columns for Postgres.
fn serialize_tool_params(
    tool_params: Option<&ToolCallConfigDatabaseInsert>,
) -> Result<SerializedToolParams, Error> {
    match tool_params {
        Some(tp) => {
            // Serialize the full tool_params for legacy compatibility
            let tool_params_json = serde_json::to_value(tp).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to serialize tool_params: {e}"),
                })
            })?;

            let dynamic_tools = serde_json::to_value(&tp.dynamic_tools).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to serialize dynamic_tools: {e}"),
                })
            })?;

            let dynamic_provider_tools =
                serde_json::to_value(&tp.dynamic_provider_tools).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!("Failed to serialize dynamic_provider_tools: {e}"),
                    })
                })?;

            let allowed_tools = serde_json::to_value(&tp.allowed_tools).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to serialize allowed_tools: {e}"),
                })
            })?;

            let tool_choice = serde_json::to_string(&tp.tool_choice).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to serialize tool_choice: {e}"),
                })
            })?;

            Ok(SerializedToolParams {
                tool_params_json,
                dynamic_tools,
                dynamic_provider_tools,
                allowed_tools: Some(allowed_tools),
                tool_choice: Some(tool_choice),
                parallel_tool_calls: tp.parallel_tool_calls,
            })
        }
        None => {
            // No tool params - use empty/null values
            Ok(SerializedToolParams {
                tool_params_json: serde_json::json!({}),
                dynamic_tools: serde_json::json!([]),
                dynamic_provider_tools: serde_json::json!([]),
                allowed_tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uuid_v7_to_timestamp() {
        // Create a known UUIDv7 with a specific timestamp
        // UUIDv7: 018f0000-0000-7000-8000-000000000000
        // This represents timestamp 1704067200000 ms (2024-01-01 00:00:00 UTC)
        let uuid = Uuid::parse_str("018f0000-0000-7000-8000-000000000000").unwrap();
        let timestamp = uuid_v7_to_timestamp(uuid);

        // The timestamp should be close to 2024-01-01
        // Note: This is an approximation since we're using a manually constructed UUID
        assert!(timestamp.timestamp() > 1700000000); // After Nov 2023
    }

    #[test]
    fn test_serialize_tool_params_none() {
        let SerializedToolParams {
            tool_params_json,
            dynamic_tools,
            dynamic_provider_tools,
            allowed_tools,
            tool_choice,
            parallel_tool_calls,
        } = serialize_tool_params(None).unwrap();

        assert_eq!(tool_params_json, serde_json::json!({}));
        assert_eq!(dynamic_tools, serde_json::json!([]));
        assert_eq!(dynamic_provider_tools, serde_json::json!([]));
        assert!(allowed_tools.is_none());
        assert!(tool_choice.is_none());
        assert!(parallel_tool_calls.is_none());
    }
}
