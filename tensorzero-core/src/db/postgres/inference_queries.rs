//! Inference queries for Postgres.
//!
//! This module implements read and write operations for inference tables in Postgres.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::QueryBuilder;
use sqlx::Row;
use sqlx::types::Json;
use std::collections::HashMap;
use uuid::Uuid;

use crate::config::snapshot::SnapshotHash;
use crate::config::{Config, MetricConfigLevel};
use crate::db::clickhouse::query_builder::{OrderBy, OrderByTerm, OrderDirection};
use crate::db::inferences::{
    CountInferencesParams, FunctionInfo, InferenceMetadata, InferenceOutputSource,
    InferenceQueries, ListInferenceMetadataParams, ListInferencesParams, PaginationParams,
};
use crate::db::postgres::inference_filter_helpers::{MetricJoinRegistry, apply_inference_filter};
use crate::db::query_helpers::json_double_escape_string_without_quotes;
use crate::db::query_helpers::uuid_to_datetime;
use crate::endpoints::inference::InferenceParams;
use crate::endpoints::stored_inferences::v1::types::{
    FloatComparisonOperator, TagComparisonOperator, TimeComparisonOperator,
};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfigType;
use crate::inference::types::ContentBlockChatOutput;
use crate::inference::types::JsonInferenceOutput;
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::stored_input::StoredInput;
use crate::inference::types::{
    ChatInferenceDatabaseInsert, FunctionType, JsonInferenceDatabaseInsert,
};
use crate::stored_inference::{
    StoredChatInferenceDatabase, StoredInferenceDatabase, StoredJsonInference,
};
use crate::tool::ToolCallConfigDatabaseInsert;
use crate::tool::config::AllowedTools;
use crate::tool::types::{ProviderTool, Tool};
use crate::tool::wire::ToolChoice;

use super::PostgresConnectionInfo;

impl FloatComparisonOperator {
    pub(super) fn to_postgres_operator(self) -> &'static str {
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
    pub(super) fn to_postgres_operator(self) -> &'static str {
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
    pub(super) fn to_postgres_operator(self) -> &'static str {
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

        let mut results = match function_config_type {
            Some(FunctionConfigType::Chat) => {
                // Single-table query for chat inferences
                let chat_results =
                    query_chat_inferences(pool, config, params, params.limit, params.offset)
                        .await?;
                chat_results
                    .into_iter()
                    .map(StoredInferenceDatabase::Chat)
                    .collect()
            }
            Some(FunctionConfigType::Json) => {
                // Single-table query for json inferences
                let json_results =
                    query_json_inferences(pool, config, params, params.limit, params.offset)
                        .await?;
                json_results
                    .into_iter()
                    .map(StoredInferenceDatabase::Json)
                    .collect()
            }
            None => {
                // Multi-table query using UNION ALL
                query_inferences_union(pool, config, params).await?
            }
        };

        // Reverse the list for "After" pagination, we queried with inverted ordering to get results after the cursor,
        // but we want to return them in original order (most recent first).
        if matches!(params.pagination, Some(PaginationParams::After { .. })) {
            results.reverse();
        }

        Ok(results)
    }

    async fn list_inference_metadata(
        &self,
        params: &ListInferenceMetadataParams,
    ) -> Result<Vec<InferenceMetadata>, Error> {
        let pool = self.get_pool_result()?;

        // Use UNION ALL to query both tables in a single query
        // Determine ORDER BY direction based on pagination
        let order_direction = match &params.pagination {
            Some(PaginationParams::After { .. }) => "ASC",
            Some(PaginationParams::Before { .. }) | None => "DESC",
        };

        let order_by_clause = format!("ORDER BY id {order_direction}");

        // Build the entire query using a single QueryBuilder
        let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
            r"
        SELECT * FROM (
            (SELECT
                'chat'::text as function_type,
                id,
                function_name,
                variant_name,
                episode_id,
                snapshot_hash
            FROM tensorzero.chat_inferences
            WHERE 1=1",
        );

        // Add chat filters
        apply_metadata_filters(&mut query_builder, params);

        query_builder.push(" ");
        query_builder.push(&order_by_clause);
        query_builder.push(" LIMIT ");
        query_builder.push_bind(params.limit as i64);

        // UNION ALL with json subquery
        query_builder.push(
            r")
            UNION ALL
            (SELECT
                'json'::text as function_type,
                id,
                function_name,
                variant_name,
                episode_id,
                snapshot_hash
            FROM tensorzero.json_inferences
            WHERE 1=1",
        );

        // Add json filters
        apply_metadata_filters(&mut query_builder, params);

        query_builder.push(" ");
        query_builder.push(&order_by_clause);
        query_builder.push(" LIMIT ");
        query_builder.push_bind(params.limit as i64);

        // Close subqueries and add outer ORDER BY + LIMIT
        query_builder.push(
            ")
        ) AS combined
        ",
        );
        query_builder.push(&order_by_clause);
        query_builder.push(" LIMIT ");
        query_builder.push_bind(params.limit as i64);

        let mut results: Vec<InferenceMetadata> =
            query_builder.build_query_as().fetch_all(pool).await?;

        // Reverse for "After" pagination
        if matches!(params.pagination, Some(PaginationParams::After { .. })) {
            results.reverse();
        }

        Ok(results)
    }

    async fn count_inferences(
        &self,
        config: &Config,
        params: &CountInferencesParams<'_>,
    ) -> Result<u64, Error> {
        let pool = self.get_pool_result()?;

        // Determine which table(s) to query based on function_name
        let function_config_type = match params.function_name {
            Some(fn_name) => Some(config.get_function(fn_name)?.config_type()),
            None => None,
        };

        let count = match function_config_type {
            Some(FunctionConfigType::Chat) => {
                count_single_table_inferences(pool, config, params, "tensorzero.chat_inferences")
                    .await?
            }
            Some(FunctionConfigType::Json) => {
                count_single_table_inferences(pool, config, params, "tensorzero.json_inferences")
                    .await?
            }
            None => {
                // Use UNION ALL to count both tables in a single query
                count_inferences_union(pool, config, params).await?
            }
        };

        Ok(count)
    }

    async fn get_function_info(
        &self,
        target_id: &Uuid,
        level: MetricConfigLevel,
    ) -> Result<Option<FunctionInfo>, Error> {
        let pool = self.get_pool_result()?;

        // Use a single UNION query to search both tables
        let mut query_builder: QueryBuilder<sqlx::Postgres> = match level {
            MetricConfigLevel::Inference => {
                let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
                    r"
                    SELECT
                        function_name,
                        variant_name,
                        episode_id,
                        'chat' as function_type
                    FROM tensorzero.chat_inferences
                    WHERE id = ",
                );
                qb.push_bind(*target_id);
                qb.push(
                    r"
                    UNION ALL
                    SELECT
                        function_name,
                        variant_name,
                        episode_id,
                        'json' as function_type
                    FROM tensorzero.json_inferences
                    WHERE id = ",
                );
                qb.push_bind(*target_id);
                qb.push(" LIMIT 1");
                qb
            }
            MetricConfigLevel::Episode => {
                let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
                    r"
                    SELECT
                        function_name,
                        variant_name,
                        episode_id,
                        function_type
                    FROM (
                        SELECT function_name, variant_name, episode_id, created_at, 'chat' as function_type
                        FROM tensorzero.chat_inferences
                        WHERE episode_id = ",
                );
                qb.push_bind(*target_id);
                qb.push(
                    r"
                        UNION ALL
                        SELECT function_name, variant_name, episode_id, created_at, 'json' as function_type
                        FROM tensorzero.json_inferences
                        WHERE episode_id = ",
                );
                qb.push_bind(*target_id);
                qb.push(
                    r"
                    ) combined
                    ORDER BY created_at DESC
                    LIMIT 1",
                );
                qb
            }
        };

        let result: Option<FunctionInfo> =
            query_builder.build_query_as().fetch_optional(pool).await?;

        Ok(result)
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
        .await?;

        match result {
            Some(row) => {
                // Deserialize the tool params from the stored JSON columns
                let dynamic_tools: Vec<Tool> = serde_json::from_value(row.dynamic_tools)?;
                let dynamic_provider_tools: Vec<ProviderTool> =
                    serde_json::from_value(row.dynamic_provider_tools)?;
                let allowed_tools: Option<AllowedTools> =
                    row.allowed_tools.map(serde_json::from_value).transpose()?;
                let tool_choice: Option<ToolChoice> =
                    row.tool_choice.map(serde_json::from_value).transpose()?;

                let tool_params = ToolCallConfigDatabaseInsert::from_stored_values(
                    dynamic_tools,
                    dynamic_provider_tools,
                    allowed_tools,
                    tool_choice,
                    row.parallel_tool_calls,
                );
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
        .await?;

        Ok(result.map(|r| r.output_schema))
    }

    // TODO(#5691): Change this to return either a Value of a typed output.
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
                .await?;

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
                .await?;

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

        // Pre-compute timestamps to propagate errors before entering push_values
        let timestamps: Vec<DateTime<Utc>> = rows
            .iter()
            .map(|row| uuid_to_datetime(row.id))
            .collect::<Result<_, _>>()?;

        let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
            r"
            INSERT INTO tensorzero.chat_inferences (
                id, function_name, variant_name, episode_id, input, output,
                inference_params, processing_time_ms, ttft_ms,
                tags, extra_body, dynamic_tools, dynamic_provider_tools,
                allowed_tools, tool_choice, parallel_tool_calls, snapshot_hash, created_at
            ) ",
        );

        query_builder.push_values(rows.iter().zip(&timestamps), |mut b, (row, created_at)| {
            let snapshot_hash_bytes: Option<Vec<u8>> =
                row.snapshot_hash.as_ref().map(|h| h.as_bytes().to_vec());

            // For tool params, use empty defaults when None
            let tool_params_ref = row.tool_params.as_ref();
            let empty_tools: Vec<Tool> = vec![];
            let empty_provider_tools: Vec<ProviderTool> = vec![];

            b.push_bind(row.id)
                .push_bind(&row.function_name)
                .push_bind(&row.variant_name)
                .push_bind(row.episode_id)
                .push_bind(Json::from(&row.input))
                .push_bind(Json::from(&row.output))
                .push_bind(Json::from(&row.inference_params))
                .push_bind(row.processing_time_ms.map(|v| v as i32))
                .push_bind(row.ttft_ms.map(|v| v as i32))
                .push_bind(Json::from(&row.tags))
                .push_bind(Json::from(&row.extra_body))
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
                .push_bind(snapshot_hash_bytes)
                .push_bind(created_at);
        });

        query_builder.build().execute(pool).await?;

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

        // Pre-compute timestamps to propagate errors before entering push_values
        let timestamps: Vec<DateTime<Utc>> = rows
            .iter()
            .map(|row| uuid_to_datetime(row.id))
            .collect::<Result<_, _>>()?;

        let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
            r"
            INSERT INTO tensorzero.json_inferences (
                id, function_name, variant_name, episode_id, input, output,
                output_schema, inference_params, processing_time_ms, ttft_ms,
                tags, extra_body, auxiliary_content, snapshot_hash, created_at
            ) ",
        );

        query_builder.push_values(rows.iter().zip(&timestamps), |mut b, (row, created_at)| {
            let snapshot_hash_bytes: Option<Vec<u8>> =
                row.snapshot_hash.as_ref().map(|h| h.as_bytes().to_vec());

            b.push_bind(row.id)
                .push_bind(&row.function_name)
                .push_bind(&row.variant_name)
                .push_bind(row.episode_id)
                .push_bind(Json::from(&row.input))
                .push_bind(Json::from(&row.output))
                .push_bind(&row.output_schema)
                .push_bind(Json::from(&row.inference_params))
                .push_bind(row.processing_time_ms.map(|v| v as i32))
                .push_bind(row.ttft_ms.map(|v| v as i32))
                .push_bind(Json::from(&row.tags))
                .push_bind(Json::from(&row.extra_body))
                .push_bind(Json::from(&row.auxiliary_content))
                .push_bind(snapshot_hash_bytes)
                .push_bind(created_at);
        });

        query_builder.build().execute(pool).await?;

        Ok(())
    }
}

// ===== Helper types =====

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
    limit: u32,
    offset: u32,
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
    query_builder.push_bind(limit as i64);

    // Add OFFSET
    query_builder.push(" OFFSET ");
    query_builder.push_bind(offset as i64);

    let results: Vec<StoredChatInferenceDatabase> =
        query_builder.build_query_as().fetch_all(pool).await?;

    Ok(results)
}

async fn query_json_inferences(
    pool: &sqlx::PgPool,
    config: &Config,
    params: &ListInferencesParams<'_>,
    limit: u32,
    offset: u32,
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
    query_builder.push_bind(limit as i64);

    // Add OFFSET
    query_builder.push(" OFFSET ");
    query_builder.push_bind(offset as i64);

    let results: Vec<StoredJsonInference> = query_builder.build_query_as().fetch_all(pool).await?;

    Ok(results)
}

// ===== UNION ALL query for multi-table inference queries =====

/// Manual implementation of FromRow for StoredInferenceDatabase to handle UNION ALL queries.
/// Uses the `inference_type` column ('chat' or 'json') to determine which variant to construct.
impl<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> for StoredInferenceDatabase {
    fn from_row(row: &'r sqlx::postgres::PgRow) -> Result<Self, sqlx::Error> {
        let inference_type: String = row.try_get("inference_type")?;

        match inference_type.as_str() {
            "chat" => {
                let chat = StoredChatInferenceDatabase::from_row(row)?;
                Ok(StoredInferenceDatabase::Chat(chat))
            }
            "json" => {
                let json = StoredJsonInference::from_row(row)?;
                Ok(StoredInferenceDatabase::Json(json))
            }
            _ => Err(sqlx::Error::ColumnDecode {
                index: "inference_type".to_string(),
                source: Box::new(Error::new(ErrorDetails::PostgresResult {
                    result_type: "inference",
                    message: format!("Unknown inference type: {inference_type}"),
                })),
            }),
        }
    }
}

/// Manual implementation of FromRow for StoredChatInferenceDatabase.
/// This allows direct deserialization from Postgres rows without an intermediate struct.
impl<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> for StoredChatInferenceDatabase {
    fn from_row(row: &'r sqlx::postgres::PgRow) -> Result<Self, sqlx::Error> {
        let id: Uuid = row.try_get("id")?;
        let function_name: String = row.try_get("function_name")?;
        let variant_name: String = row.try_get("variant_name")?;
        let episode_id: Uuid = row.try_get("episode_id")?;
        let timestamp: DateTime<Utc> = row.try_get("timestamp")?;
        let input: Json<StoredInput> = row.try_get("input")?;
        let output: Json<Vec<ContentBlockChatOutput>> = row.try_get("output")?;
        let dispreferred_output: Option<Json<Vec<ContentBlockChatOutput>>> =
            row.try_get("dispreferred_output")?;
        let tags: Json<HashMap<String, String>> = row.try_get("tags")?;
        let extra_body: Json<UnfilteredInferenceExtraBody> = row.try_get("extra_body")?;
        let inference_params: Json<InferenceParams> = row.try_get("inference_params")?;
        let processing_time_ms: Option<i32> = row.try_get("processing_time_ms")?;
        let ttft_ms: Option<i32> = row.try_get("ttft_ms")?;

        // Get chat-specific fields for tool_params reconstruction
        let dynamic_tools: Vec<Tool> = row.try_get::<Json<Vec<Tool>>, _>("dynamic_tools")?.0;
        let dynamic_provider_tools: Vec<ProviderTool> = row
            .try_get::<Json<Vec<ProviderTool>>, _>("dynamic_provider_tools")?
            .0;
        let allowed_tools: Option<AllowedTools> = row
            .try_get::<Option<Json<AllowedTools>>, _>("allowed_tools")?
            .map(|v| v.0);
        let tool_choice: Option<ToolChoice> = row
            .try_get::<Option<Json<ToolChoice>>, _>("tool_choice")?
            .map(|v| v.0);
        let parallel_tool_calls: Option<bool> = row.try_get("parallel_tool_calls")?;

        let tool_params = ToolCallConfigDatabaseInsert::from_stored_values(
            dynamic_tools,
            dynamic_provider_tools,
            allowed_tools,
            tool_choice,
            parallel_tool_calls,
        )
        .unwrap_or_default();

        let dispreferred_outputs = dispreferred_output.map(|d| vec![d.0]).unwrap_or_default();

        Ok(StoredChatInferenceDatabase {
            function_name,
            variant_name,
            input: input.0,
            output: output.0,
            dispreferred_outputs,
            timestamp,
            episode_id,
            inference_id: id,
            tool_params,
            tags: tags.0,
            extra_body: extra_body.0,
            inference_params: inference_params.0,
            processing_time_ms: processing_time_ms.map(|v| v as u64),
            ttft_ms: ttft_ms.map(|v| v as u64),
        })
    }
}

/// Manual implementation of FromRow for StoredJsonInference.
/// This allows direct deserialization from Postgres rows without an intermediate struct.
impl<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> for StoredJsonInference {
    fn from_row(row: &'r sqlx::postgres::PgRow) -> Result<Self, sqlx::Error> {
        let id: Uuid = row.try_get("id")?;
        let function_name: String = row.try_get("function_name")?;
        let variant_name: String = row.try_get("variant_name")?;
        let episode_id: Uuid = row.try_get("episode_id")?;
        let timestamp: DateTime<Utc> = row.try_get("timestamp")?;
        let input: Json<StoredInput> = row.try_get("input")?;
        let output: Json<JsonInferenceOutput> = row.try_get("output")?;
        let dispreferred_output: Option<Json<JsonInferenceOutput>> =
            row.try_get("dispreferred_output")?;
        let output_schema: Value = row.try_get("output_schema")?;
        let tags: Json<HashMap<String, String>> = row.try_get("tags")?;
        let extra_body: Json<UnfilteredInferenceExtraBody> = row.try_get("extra_body")?;
        let inference_params: Json<InferenceParams> = row.try_get("inference_params")?;
        let processing_time_ms: Option<i32> = row.try_get("processing_time_ms")?;
        let ttft_ms: Option<i32> = row.try_get("ttft_ms")?;

        let dispreferred_outputs = dispreferred_output.map(|d| vec![d.0]).unwrap_or_default();

        Ok(StoredJsonInference {
            function_name,
            variant_name,
            input: input.0,
            output: output.0,
            dispreferred_outputs,
            timestamp,
            episode_id,
            inference_id: id,
            output_schema,
            tags: tags.0,
            extra_body: extra_body.0,
            inference_params: inference_params.0,
            processing_time_ms: processing_time_ms.map(|v| v as u64),
            ttft_ms: ttft_ms.map(|v| v as u64),
        })
    }
}

/// Validates ORDER BY terms for UNION ALL queries (when function_name is not specified).
/// Returns an error if ORDER BY metric or search_relevance is used, as these are not supported
/// for cross-table queries.
fn validate_union_order_by(order_by: Option<&[OrderBy]>) -> Result<(), Error> {
    if let Some(order_by) = order_by {
        for o in order_by {
            if matches!(o.term, OrderByTerm::Metric { .. }) {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "ORDER BY metric is not supported when querying without function_name"
                        .to_string(),
                }));
            }
            if matches!(o.term, OrderByTerm::SearchRelevance) {
                return Err(Error::new(ErrorDetails::PostgresQuery {
                    message:
                        "ORDER BY search_relevance is not yet supported for Postgres inference queries"
                            .to_string(),
                }));
            }
        }
    }
    Ok(())
}

/// Builds a query for inferences from both tables using UNION ALL.
/// Returns a QueryBuilder that can be executed against a Postgres connection.
fn build_inferences_union_query(
    config: &Config,
    params: &ListInferencesParams<'_>,
) -> Result<QueryBuilder<sqlx::Postgres>, Error> {
    // For UNION ALL, we don't support ORDER BY metric since each subquery would need its own JOINs
    // and the metric values need to be consistent across both tables
    validate_union_order_by(params.order_by)?;

    // Build ORDER BY clause for the inner and outer queries
    let should_invert_directions =
        matches!(params.pagination, Some(PaginationParams::After { .. }));

    let id_direction = if let Some(ref pagination) = params.pagination {
        match pagination {
            PaginationParams::After { .. } => OrderDirection::Asc,
            PaginationParams::Before { .. } => OrderDirection::Desc,
        }
    } else {
        OrderDirection::Desc // Default: most recent first
    };

    let mut order_clauses = Vec::new();

    // Add user-specified ordering if present
    // Only Timestamp is allowed here; Metric and SearchRelevance are rejected above
    if let Some(order_by) = params.order_by {
        for o in order_by {
            let effective_direction = if should_invert_directions {
                o.direction.inverted()
            } else {
                o.direction
            };
            order_clauses.push(format!(
                "timestamp {} NULLS LAST",
                effective_direction.to_clickhouse_direction()
            ));
        }
    }

    // Always add id as tie-breaker for deterministic ordering
    order_clauses.push(format!("id {}", id_direction.to_clickhouse_direction()));

    let order_by_clause = format!("ORDER BY {}", order_clauses.join(", "));

    // Inner limit is (limit + offset) to ensure we fetch enough rows before applying outer offset
    let inner_limit = (params.limit + params.offset) as i64;

    // Build the SELECT clause based on output_source
    let (chat_output_select, json_output_select) = match params.output_source {
        InferenceOutputSource::None | InferenceOutputSource::Inference => (
            "i.output, NULL::jsonb as dispreferred_output",
            "i.output, NULL::jsonb as dispreferred_output",
        ),
        InferenceOutputSource::Demonstration => (
            "demo_f.value AS output, i.output as dispreferred_output",
            "demo_f.value AS output, i.output as dispreferred_output",
        ),
    };

    // Demonstration JOIN clause
    let demo_join = if params.output_source == InferenceOutputSource::Demonstration {
        r"
        JOIN (
            SELECT DISTINCT ON (inference_id)
                inference_id,
                value
            FROM tensorzero.demonstration_feedback
            ORDER BY inference_id, created_at DESC
        ) AS demo_f ON i.id = demo_f.inference_id
        "
    } else {
        ""
    };

    // Build the entire query using a single QueryBuilder to properly handle bind parameters
    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(format!(
        r"
        SELECT * FROM (
            (SELECT
                'chat'::text as inference_type,
                i.id,
                i.function_name,
                i.variant_name,
                i.episode_id,
                i.created_at as timestamp,
                i.input,
                {chat_output_select},
                i.dynamic_tools,
                i.dynamic_provider_tools,
                i.allowed_tools,
                i.tool_choice,
                i.parallel_tool_calls,
                NULL::jsonb as output_schema,
                i.tags,
                i.extra_body,
                i.inference_params,
                i.processing_time_ms,
                i.ttft_ms
            FROM tensorzero.chat_inferences i
            {demo_join}
            WHERE 1=1"
    ));

    // Add chat filters
    apply_union_filters(&mut query_builder, config, params)?;

    query_builder.push(" ");
    query_builder.push(&order_by_clause);
    query_builder.push(" LIMIT ");
    query_builder.push_bind(inner_limit);

    // UNION ALL with json subquery
    query_builder.push(format!(
        r")
            UNION ALL
            (SELECT
                'json'::text as inference_type,
                i.id,
                i.function_name,
                i.variant_name,
                i.episode_id,
                i.created_at as timestamp,
                i.input,
                {json_output_select},
                NULL::jsonb as dynamic_tools,
                NULL::jsonb as dynamic_provider_tools,
                NULL::jsonb as allowed_tools,
                NULL::jsonb as tool_choice,
                NULL::boolean as parallel_tool_calls,
                i.output_schema,
                i.tags,
                i.extra_body,
                i.inference_params,
                i.processing_time_ms,
                i.ttft_ms
            FROM tensorzero.json_inferences i
            {demo_join}
            WHERE 1=1"
    ));

    // Add json filters
    apply_union_filters(&mut query_builder, config, params)?;

    query_builder.push(" ");
    query_builder.push(&order_by_clause);
    query_builder.push(" LIMIT ");
    query_builder.push_bind(inner_limit);

    // Close subqueries and add outer ORDER BY + LIMIT + OFFSET
    query_builder.push(
        ")
        ) AS combined
        ",
    );
    query_builder.push(&order_by_clause);
    query_builder.push(" LIMIT ");
    query_builder.push_bind(params.limit as i64);
    query_builder.push(" OFFSET ");
    query_builder.push_bind(params.offset as i64);

    Ok(query_builder)
}

/// Query inferences from both tables using UNION ALL.
/// This pushes sorting and pagination to the database.
async fn query_inferences_union(
    pool: &sqlx::PgPool,
    config: &Config,
    params: &ListInferencesParams<'_>,
) -> Result<Vec<StoredInferenceDatabase>, Error> {
    let mut query_builder = build_inferences_union_query(config, params)?;

    let results: Vec<StoredInferenceDatabase> =
        query_builder.build_query_as().fetch_all(pool).await?;

    Ok(results)
}

/// Apply filters common to both chat and json subqueries in UNION ALL.
fn apply_union_filters(
    query_builder: &mut QueryBuilder<sqlx::Postgres>,
    config: &Config,
    params: &ListInferencesParams<'_>,
) -> Result<(), Error> {
    // Note: function_name filter is not applied here since UNION ALL queries
    // don't have function_name specified (that's the whole point)

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
    apply_inference_filter(query_builder, params.filters, config)?;

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

    Ok(())
}

// ===== Helper functions for inference metadata queries =====

/// Manual implementation of FromRow for InferenceMetadata to handle UNION ALL queries.
/// Converts `snapshot_hash` bytes to string via SnapshotHash.
///
/// TODO(shuyangli): come up with a nice way to do the snapshot_hash conversion so we don't
/// need to manually impl sqlx::FromRow. Because the origin type is Vec<u8> and the destination
/// type is String, we can't/shouldn't directly impl TryFrom<Vec<u8>> for String.
impl<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> for InferenceMetadata {
    fn from_row(row: &'r sqlx::postgres::PgRow) -> Result<Self, sqlx::Error> {
        let snapshot_hash_bytes: Option<Vec<u8>> = row.try_get("snapshot_hash")?;
        let snapshot_hash =
            snapshot_hash_bytes.map(|bytes| SnapshotHash::from_bytes(&bytes).to_string());

        Ok(InferenceMetadata {
            id: row.try_get("id")?,
            function_name: row.try_get("function_name")?,
            variant_name: row.try_get("variant_name")?,
            episode_id: row.try_get("episode_id")?,
            function_type: row.try_get("function_type")?,
            snapshot_hash,
        })
    }
}

/// Apply filters for metadata queries.
fn apply_metadata_filters(
    query_builder: &mut QueryBuilder<sqlx::Postgres>,
    params: &ListInferenceMetadataParams,
) {
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

    // Handle pagination cursor
    match &params.pagination {
        Some(PaginationParams::Before { id }) => {
            query_builder.push(" AND id < ");
            query_builder.push_bind(*id);
        }
        Some(PaginationParams::After { id }) => {
            query_builder.push(" AND id > ");
            query_builder.push_bind(*id);
        }
        None => {}
    }
}

// ===== Helper functions for count queries =====

/// Count inferences from a single table (chat or json).
async fn count_single_table_inferences(
    pool: &sqlx::PgPool,
    config: &Config,
    params: &CountInferencesParams<'_>,
    table_name: &str,
) -> Result<u64, Error> {
    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(format!(
        r"
        SELECT COUNT(*)::BIGINT
        FROM {table_name} i
        WHERE 1=1
        "
    ));

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

    // Apply inference filter (e.g., DemonstrationFeedback, metric filters, etc.)
    apply_inference_filter(&mut query_builder, params.filters, config)?;

    // Apply search query filter
    if let Some(search_query) = params.search_query_experimental {
        let json_escaped_text_query = json_double_escape_string_without_quotes(search_query)?;
        let search_pattern = format!("%{json_escaped_text_query}%");
        query_builder.push(" AND (i.input::text ILIKE ");
        query_builder.push_bind(search_pattern.clone());
        query_builder.push(" OR i.output::text ILIKE ");
        query_builder.push_bind(search_pattern);
        query_builder.push(")");
    }

    let query = query_builder.build_query_scalar::<i64>();
    let count: i64 = query.fetch_one(pool).await?;

    Ok(count as u64)
}

/// Count inferences from both tables using UNION ALL.
async fn count_inferences_union(
    pool: &sqlx::PgPool,
    config: &Config,
    params: &CountInferencesParams<'_>,
) -> Result<u64, Error> {
    // Build the entire query using a single QueryBuilder
    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
        r"
        SELECT COUNT(*)::BIGINT FROM (
            (SELECT i.id FROM tensorzero.chat_inferences i WHERE 1=1",
    );

    // Add chat filters (no function_name filter since we're querying both tables)
    apply_count_filters(&mut query_builder, config, params)?;

    // UNION ALL with json subquery
    query_builder.push(
        r")
            UNION ALL
            (SELECT i.id FROM tensorzero.json_inferences i WHERE 1=1",
    );

    // Add json filters
    apply_count_filters(&mut query_builder, config, params)?;

    // Close subqueries
    query_builder.push(
        ")
        ) AS combined
    ",
    );

    let query = query_builder.build_query_scalar::<i64>();
    let count: i64 = query.fetch_one(pool).await?;

    Ok(count as u64)
}

/// Apply filters for count queries.
fn apply_count_filters(
    query_builder: &mut QueryBuilder<sqlx::Postgres>,
    config: &Config,
    params: &CountInferencesParams<'_>,
) -> Result<(), Error> {
    // Note: function_name filter is not applied here for UNION ALL queries
    // since that's used to determine which tables to query

    if let Some(variant_name) = params.variant_name {
        query_builder.push(" AND i.variant_name = ");
        query_builder.push_bind(variant_name);
    }

    if let Some(episode_id) = params.episode_id {
        query_builder.push(" AND i.episode_id = ");
        query_builder.push_bind(*episode_id);
    }

    // Apply inference filter (e.g., DemonstrationFeedback, metric filters, etc.)
    apply_inference_filter(query_builder, params.filters, config)?;

    // Apply search query filter
    if let Some(search_query) = params.search_query_experimental {
        let json_escaped_text_query = json_double_escape_string_without_quotes(search_query)?;
        let search_pattern = format!("%{json_escaped_text_query}%");
        query_builder.push(" AND (i.input::text ILIKE ");
        query_builder.push_bind(search_pattern.clone());
        query_builder.push(" OR i.output::text ILIKE ");
        query_builder.push_bind(search_pattern);
        query_builder.push(")");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::config::ConfigFileGlob;
    use crate::db::clickhouse::query_builder::{OrderBy, OrderByTerm, OrderDirection};
    use crate::db::inferences::ListInferencesParams;
    use crate::db::test_helpers::{assert_query_contains, assert_query_equals};

    async fn get_test_config() -> Config {
        Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(Path::new("tests/e2e/config/tensorzero.*.toml"))
                .unwrap(),
            false,
        )
        .await
        .unwrap()
        .into_config_without_writing_for_tests()
    }

    #[tokio::test]
    async fn test_build_inferences_union_query_allows_timestamp_order_by() {
        let config = get_test_config().await;
        let order_by = vec![OrderBy {
            term: OrderByTerm::Timestamp,
            direction: OrderDirection::Desc,
        }];
        let params = ListInferencesParams {
            order_by: Some(&order_by),
            ..Default::default()
        };

        let result = build_inferences_union_query(&config, &params);
        assert!(
            result.is_ok(),
            "ORDER BY timestamp should be allowed: {:?}",
            result.err()
        );

        let query_builder = result.unwrap();
        let sql_str = query_builder.sql();
        let sql = sql_str.as_str();
        assert_query_contains(sql, "ORDER BY timestamp DESC");
    }

    #[tokio::test]
    async fn test_build_inferences_union_query_allows_no_order_by() {
        let config = get_test_config().await;
        let params = ListInferencesParams::default();

        let result = build_inferences_union_query(&config, &params);
        assert!(
            result.is_ok(),
            "No ORDER BY should be allowed: {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn test_build_inferences_union_query_rejects_metric_order_by() {
        let config = get_test_config().await;
        let order_by = vec![OrderBy {
            term: OrderByTerm::Metric {
                name: "test_metric".to_string(),
            },
            direction: OrderDirection::Desc,
        }];
        let params = ListInferencesParams {
            order_by: Some(&order_by),
            ..Default::default()
        };

        let result = build_inferences_union_query(&config, &params);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("ORDER BY metric should be rejected"),
        };
        assert!(
            err.to_string()
                .contains("ORDER BY metric is not supported when querying without function_name"),
            "Error message should mention metric not supported: {err}"
        );
    }

    #[tokio::test]
    async fn test_build_inferences_union_query_rejects_search_relevance_order_by() {
        let config = get_test_config().await;
        let order_by = vec![OrderBy {
            term: OrderByTerm::SearchRelevance,
            direction: OrderDirection::Desc,
        }];
        let params = ListInferencesParams {
            order_by: Some(&order_by),
            ..Default::default()
        };

        let result = build_inferences_union_query(&config, &params);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("ORDER BY search_relevance should be rejected"),
        };
        assert!(
            err.to_string()
                .contains("ORDER BY search_relevance is not yet supported"),
            "Error message should mention search_relevance not supported: {err}"
        );
    }

    #[tokio::test]
    async fn test_build_inferences_union_query_rejects_metric_in_multiple_order_by_terms() {
        let config = get_test_config().await;
        let order_by = vec![
            OrderBy {
                term: OrderByTerm::Timestamp,
                direction: OrderDirection::Desc,
            },
            OrderBy {
                term: OrderByTerm::Metric {
                    name: "test_metric".to_string(),
                },
                direction: OrderDirection::Asc,
            },
        ];
        let params = ListInferencesParams {
            order_by: Some(&order_by),
            ..Default::default()
        };

        let result = build_inferences_union_query(&config, &params);
        assert!(
            result.is_err(),
            "ORDER BY with metric in multiple terms should be rejected"
        );
    }

    #[tokio::test]
    async fn test_build_inferences_union_query_generates_union_all() {
        let config = get_test_config().await;
        let params = ListInferencesParams::default();

        let result = build_inferences_union_query(&config, &params);
        assert!(result.is_ok(), "Query building should succeed");

        let query_builder = result.unwrap();
        let sql_str = query_builder.sql();
        let sql = sql_str.as_str();

        let expected = r"
            SELECT * FROM (
                (SELECT
                    'chat'::text as inference_type,
                    i.id,
                    i.function_name,
                    i.variant_name,
                    i.episode_id,
                    i.created_at as timestamp,
                    i.input,
                    i.output, NULL::jsonb as dispreferred_output,
                    i.dynamic_tools,
                    i.dynamic_provider_tools,
                    i.allowed_tools,
                    i.tool_choice,
                    i.parallel_tool_calls,
                    NULL::jsonb as output_schema,
                    i.tags,
                    i.extra_body,
                    i.inference_params,
                    i.processing_time_ms,
                    i.ttft_ms
                FROM tensorzero.chat_inferences i
                WHERE 1=1 ORDER BY id DESC LIMIT $1)
                UNION ALL
                (SELECT
                    'json'::text as inference_type,
                    i.id,
                    i.function_name,
                    i.variant_name,
                    i.episode_id,
                    i.created_at as timestamp,
                    i.input,
                    i.output, NULL::jsonb as dispreferred_output,
                    NULL::jsonb as dynamic_tools,
                    NULL::jsonb as dynamic_provider_tools,
                    NULL::jsonb as allowed_tools,
                    NULL::jsonb as tool_choice,
                    NULL::boolean as parallel_tool_calls,
                    i.output_schema,
                    i.tags,
                    i.extra_body,
                    i.inference_params,
                    i.processing_time_ms,
                    i.ttft_ms
                FROM tensorzero.json_inferences i
                WHERE 1=1 ORDER BY id DESC LIMIT $2)
            ) AS combined
            ORDER BY id DESC LIMIT $3 OFFSET $4
        ";

        assert_query_equals(sql, expected);
    }
}
