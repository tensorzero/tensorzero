use async_trait::async_trait;
use itertools::Itertools;
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

use crate::config::{
    Config, MetricConfig, MetricConfigLevel, MetricConfigOptimize, MetricConfigType,
};
use crate::db::TimeWindow;
use crate::db::clickhouse::query_builder::parameters::add_parameter;
use crate::db::clickhouse::query_builder::{
    ClickhouseType, JoinRegistry, OrderByTerm, OrderDirection, QueryParameter,
    generate_order_by_sql,
};
use crate::db::clickhouse::select_queries::parse_count;
use crate::db::clickhouse::{ClickHouseConnectionInfo, TableName};
use crate::db::inferences::{
    ClickHouseStoredInferenceWithDispreferredOutputs, CountByVariant,
    CountInferencesForFunctionParams, CountInferencesParams,
    CountInferencesWithDemonstrationFeedbacksParams, CountInferencesWithFeedbackParams,
    DEFAULT_INFERENCE_QUERY_LIMIT, FunctionInferenceCount, FunctionInfo,
    GetFunctionThroughputByVariantParams, InferenceMetadata, InferenceOutputSource,
    InferenceQueries, ListInferenceMetadataParams, ListInferencesParams, PaginationParams,
    VariantThroughput,
};
use crate::db::query_helpers::json_double_escape_string_without_quotes;
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfigType;
use crate::inference::types::{ChatInferenceDatabaseInsert, JsonInferenceDatabaseInsert};
use crate::stored_inference::StoredInferenceDatabase;
use crate::tool::ToolCallConfigDatabaseInsert;
use crate::tool::deserialize_optional_tool_info;
use serde::Deserialize;
use serde_json::Value;

/// Represents the structured parts of a single-table query
/// This allows the caller to insert JOINs between the SELECT/FROM and WHERE clauses
struct SingleTableQuery {
    /// The SELECT and FROM clauses (e.g., "SELECT ... FROM table AS i")
    select_from_sql_fragment: String,
    /// The WHERE clause if present (e.g., "WHERE condition1 AND condition2"), or empty string
    where_sql_fragment: String,
}

#[async_trait]
impl InferenceQueries for ClickHouseConnectionInfo {
    async fn list_inferences(
        &self,
        config: &Config,
        params: &ListInferencesParams<'_>,
    ) -> Result<Vec<StoredInferenceDatabase>, Error> {
        let (sql, bound_parameters) = generate_list_inferences_sql(config, params)?;
        let query_params = bound_parameters
            .iter()
            .map(|p| (p.name.as_str(), p.value.as_str()))
            .collect();
        let response = self.inner.run_query_synchronous(sql, &query_params).await?;
        let mut inferences = response
            .response
            .trim()
            .lines()
            .map(|line| {
                serde_json::from_str::<ClickHouseStoredInferenceWithDispreferredOutputs>(line)
                    .map_err(|e| {
                        Error::new(ErrorDetails::ClickHouseQuery {
                            message: format!("Failed to deserialize response: {e:?}"),
                        })
                    })
                    .and_then(ClickHouseStoredInferenceWithDispreferredOutputs::try_into)
            })
            .collect::<Result<Vec<StoredInferenceDatabase>, Error>>()?;

        // Reverse the list for "After" pagination, we queried with inverted ordering to get results after the cursor,
        // but we want to return them in original order (most recent first).
        if matches!(params.pagination, Some(PaginationParams::After { .. })) {
            inferences.reverse();
        }

        Ok(inferences)
    }

    async fn list_inference_metadata(
        &self,
        params: &ListInferenceMetadataParams,
    ) -> Result<Vec<InferenceMetadata>, Error> {
        let (sql, query_params) = generate_list_inference_metadata_sql(params);
        let query_params_refs: std::collections::HashMap<&str, &str> = query_params
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let response = self
            .inner
            .run_query_synchronous(sql, &query_params_refs)
            .await?;

        if response.response.is_empty() {
            return Ok(vec![]);
        }

        let mut results: Vec<InferenceMetadata> = response
            .response
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                serde_json::from_str(line).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: format!("Failed to parse InferenceMetadata row: {e}"),
                    })
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        // Reverse results for "after" pagination (we queried ASC, but want DESC output)
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
        let (sql, bound_parameters) = generate_count_inferences_sql(config, params)?;
        let query_params = bound_parameters
            .iter()
            .map(|p| (p.name.as_str(), p.value.as_str()))
            .collect();
        let response = self.inner.run_query_synchronous(sql, &query_params).await?;

        // Parse the count directly (no JSON format)
        let count_str = response.response.trim();
        let count: u64 = count_str.parse().map_err(|e: std::num::ParseIntError| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: format!("Failed to parse count '{count_str}': {e}"),
            })
        })?;

        Ok(count)
    }

    async fn get_function_info(
        &self,
        target_id: &Uuid,
        level: MetricConfigLevel,
    ) -> Result<Option<FunctionInfo>, Error> {
        let query = match level {
            MetricConfigLevel::Inference => {
                r"
                SELECT function_name, function_type, variant_name, episode_id
                FROM InferenceById
                WHERE id_uint = toUInt128({target_id:UUID})
                LIMIT 1
                FORMAT JSONEachRow
                SETTINGS max_threads=1
            "
            }
            MetricConfigLevel::Episode => {
                r"
                SELECT function_name, function_type, variant_name, uint_to_uuid(episode_id_uint) as episode_id
                FROM InferenceByEpisodeId
                WHERE episode_id_uint = toUInt128({target_id:UUID})
                LIMIT 1
                FORMAT JSONEachRow
                SETTINGS max_threads=1
            "
            }
        };

        let target_id_str = target_id.to_string();
        let params = HashMap::from([("target_id", target_id_str.as_str())]);
        let response = self
            .inner
            .run_query_synchronous(query.to_string(), &params)
            .await?;

        if response.response.is_empty() {
            return Ok(None);
        }

        let info: FunctionInfo = serde_json::from_str(&response.response).map_err(|e| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: e.to_string(),
            })
        })?;

        Ok(Some(info))
    }

    async fn get_chat_inference_tool_params(
        &self,
        function_name: &str,
        inference_id: Uuid,
    ) -> Result<Option<ToolCallConfigDatabaseInsert>, Error> {
        let query = r"
            SELECT tool_params, dynamic_tools, dynamic_provider_tools, allowed_tools, tool_choice
            FROM ChatInference
            WHERE function_name = {function_name:String} AND id = {inference_id:String}
            FORMAT JSONEachRow
        ";

        let inference_id_str = inference_id.to_string();
        let params = HashMap::from([
            ("function_name", function_name),
            ("inference_id", inference_id_str.as_str()),
        ]);
        let response = self
            .inner
            .run_query_synchronous(query.to_string(), &params)
            .await?;

        if response.response.is_empty() {
            return Ok(None);
        }

        #[derive(Debug, Deserialize)]
        struct ToolParamsResult {
            #[serde(flatten, deserialize_with = "deserialize_optional_tool_info")]
            tool_params: Option<ToolCallConfigDatabaseInsert>,
        }

        let result: ToolParamsResult = serde_json::from_str(&response.response).map_err(|e| {
            Error::new(ErrorDetails::ClickHouseQuery {
                message: format!("Failed to parse tool params result: {e}"),
            })
        })?;

        Ok(result.tool_params)
    }

    async fn get_json_inference_output_schema(
        &self,
        function_name: &str,
        inference_id: Uuid,
    ) -> Result<Option<Value>, Error> {
        let query = r"
            SELECT output_schema
            FROM JsonInference
            WHERE function_name = {function_name:String} AND id = {inference_id:String}
            FORMAT JSONEachRow
        ";

        let inference_id_str = inference_id.to_string();
        let params = HashMap::from([
            ("function_name", function_name),
            ("inference_id", inference_id_str.as_str()),
        ]);
        let response = self
            .inner
            .run_query_synchronous(query.to_string(), &params)
            .await?;

        if response.response.is_empty() {
            return Ok(None);
        }

        #[derive(Debug, Deserialize)]
        struct OutputSchemaResult {
            output_schema: String,
        }

        let result: OutputSchemaResult = serde_json::from_str(&response.response).map_err(|e| {
            Error::new(ErrorDetails::ClickHouseQuery {
                message: format!("Failed to parse output schema result: {e}"),
            })
        })?;

        let output_schema: Value = serde_json::from_str(&result.output_schema).map_err(|e| {
            Error::new(ErrorDetails::ClickHouseQuery {
                message: format!("Failed to parse output schema: {e}"),
            })
        })?;

        Ok(Some(output_schema))
    }

    async fn get_inference_output(
        &self,
        function_info: &FunctionInfo,
        inference_id: Uuid,
    ) -> Result<Option<String>, Error> {
        let table_name = function_info.function_type.inference_table_name();

        // Build the query with parameterized values
        let query = format!(
            r"
            SELECT output
            FROM {table_name}
            WHERE
                id = {{inference_id:String}} AND
                episode_id = {{episode_id:UUID}} AND
                function_name = {{function_name:String}} AND
                variant_name = {{variant_name:String}}
            LIMIT 1
            FORMAT JSONEachRow
            SETTINGS max_threads=1
            "
        );

        let inference_id_str = inference_id.to_string();
        let episode_id_str = function_info.episode_id.to_string();
        let params = HashMap::from([
            ("inference_id", inference_id_str.as_str()),
            ("episode_id", episode_id_str.as_str()),
            ("function_name", function_info.function_name.as_str()),
            ("variant_name", function_info.variant_name.as_str()),
        ]);

        let response = self.inner.run_query_synchronous(query, &params).await?;

        if response.response.is_empty() {
            return Ok(None);
        }

        #[derive(Debug, Deserialize)]
        struct OutputResult {
            output: String,
        }

        let result: OutputResult = serde_json::from_str(&response.response).map_err(|e| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: format!("Failed to parse output result: {e}"),
            })
        })?;

        Ok(Some(result.output))
    }

    // ===== Write methods =====

    async fn insert_chat_inferences(
        &self,
        rows: &[ChatInferenceDatabaseInsert],
    ) -> Result<(), Error> {
        if rows.is_empty() {
            return Ok(());
        }
        self.write_batched(rows, TableName::ChatInference).await
    }

    async fn insert_json_inferences(
        &self,
        rows: &[JsonInferenceDatabaseInsert],
    ) -> Result<(), Error> {
        if rows.is_empty() {
            return Ok(());
        }
        self.write_batched(rows, TableName::JsonInference).await
    }

    // ===== Inference count methods (merged from InferenceCountQueries trait) =====

    async fn count_inferences_for_function(
        &self,
        params: CountInferencesForFunctionParams<'_>,
    ) -> Result<u64, Error> {
        let (query, query_params) = build_count_inferences_query(&params);
        let response = self.run_query_synchronous(query, &query_params).await?;
        parse_count(&response.response)
    }

    async fn count_inferences_by_variant(
        &self,
        params: CountInferencesForFunctionParams<'_>,
    ) -> Result<Vec<CountByVariant>, Error> {
        let (query, query_params) = build_count_inferences_by_variant_query(&params);
        let response = self.run_query_synchronous(query, &query_params).await?;

        let result: Vec<CountByVariant> = response
            .response
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                let datapoint: Result<CountByVariant, Error> =
                    serde_json::from_str(line).map_err(|e| {
                        Error::new(ErrorDetails::ClickHouseDeserialization {
                            message: format!("Failed to deserialize CountByVariant info: {e}"),
                        })
                    });
                datapoint
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(result)
    }

    async fn count_inferences_with_feedback(
        &self,
        params: CountInferencesWithFeedbackParams<'_>,
    ) -> Result<u64, Error> {
        let (query, params_owned) = build_count_metric_feedbacks_query(
            params.function_name,
            params.function_type,
            params.metric_name,
            params.metric_config,
            params.metric_threshold,
        );
        let query_params: HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;
        parse_count(&response.response)
    }

    async fn count_inferences_with_demonstration_feedback(
        &self,
        params: CountInferencesWithDemonstrationFeedbacksParams<'_>,
    ) -> Result<u64, Error> {
        let (query, params_owned) = build_count_demonstration_feedbacks_query(params);
        let query_params: HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;
        parse_count(&response.response)
    }

    async fn count_inferences_for_episode(&self, episode_id: Uuid) -> Result<u64, Error> {
        let mut query_params_owned = HashMap::new();
        query_params_owned.insert("episode_id".to_string(), episode_id.to_string());

        let query = "SELECT COUNT() AS count
             FROM InferenceByEpisodeId FINAL
             WHERE episode_id_uint = toUInt128(toUUID({episode_id:String}))
             FORMAT JSONEachRow"
            .to_string();

        let query_params: HashMap<&str, &str> = query_params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;
        parse_count(&response.response)
    }

    async fn get_function_throughput_by_variant(
        &self,
        params: GetFunctionThroughputByVariantParams<'_>,
    ) -> Result<Vec<VariantThroughput>, Error> {
        let (query, params_owned) = build_function_throughput_by_variant_query(&params);
        let query_params: HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;

        let result: Vec<VariantThroughput> = response
            .response
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                serde_json::from_str(line).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: format!("Failed to deserialize VariantThroughput: {e}"),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(result)
    }

    async fn list_functions_with_inference_count(
        &self,
    ) -> Result<Vec<FunctionInferenceCount>, Error> {
        let query = build_list_functions_with_inference_count_query();
        let response = self.run_query_synchronous_no_params(query).await?;

        let result: Vec<FunctionInferenceCount> = response
            .response
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                serde_json::from_str(line).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: format!("Failed to deserialize FunctionInferenceCount: {e}"),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(result)
    }
}

/// Generates the SQL query and parameters for counting inferences.
pub(crate) fn generate_count_inferences_sql(
    config: &Config,
    opts: &CountInferencesParams<'_>,
) -> Result<(String, Vec<QueryParameter>), Error> {
    let mut query_params: Vec<QueryParameter> = Vec::new();
    let mut param_idx_counter = 0;

    let sql = match opts.function_name {
        // If function_name is provided, we know which table to query
        Some(function_name) => {
            let mut joins = JoinRegistry::new();
            let function_config = config.get_function(function_name)?;
            let query = generate_count_query_for_table(
                config,
                opts,
                function_config.table_name(),
                &mut joins,
                &mut query_params,
                &mut param_idx_counter,
            )?;

            // Construct final query: SELECT COUNT(*) FROM table + JOINs + WHERE
            let mut sql = query.select_from_sql_fragment;
            if !joins.get_clauses().is_empty() {
                sql.push_str(&joins.get_clauses().join("\n"));
            }
            sql.push('\n');
            sql.push_str(&query.where_sql_fragment);
            sql
        }
        None => {
            // Query both tables with UNION ALL, then count
            let mut chat_joins = JoinRegistry::new();
            let chat_query = generate_count_query_for_table(
                config,
                opts,
                "ChatInference",
                &mut chat_joins,
                &mut query_params,
                &mut param_idx_counter,
            )?;

            let mut json_joins = JoinRegistry::new();
            let json_query = generate_count_query_for_table(
                config,
                opts,
                "JsonInference",
                &mut json_joins,
                &mut query_params,
                &mut param_idx_counter,
            )?;

            // Build subqueries that select just the id for counting
            let chat_sql = format!(
                "SELECT i.id FROM ChatInference AS i {joins} {where_clause}",
                joins = chat_joins.get_clauses().join("\n"),
                where_clause = chat_query.where_sql_fragment,
            );
            let json_sql = format!(
                "SELECT i.id FROM JsonInference AS i {joins} {where_clause}",
                joins = json_joins.get_clauses().join("\n"),
                where_clause = json_query.where_sql_fragment,
            );

            format!("SELECT toUInt64(COUNT(*)) FROM ({chat_sql} UNION ALL {json_sql})")
        }
    };

    Ok((sql, query_params))
}

/// Generates WHERE clauses and JOINs for a count query on a specific table.
/// Returns a SingleTableQuery with just the WHERE part filled in.
fn generate_count_query_for_table(
    config: &Config,
    opts: &CountInferencesParams<'_>,
    table_name: &str,
    joins: &mut JoinRegistry,
    query_params: &mut Vec<QueryParameter>,
    param_idx_counter: &mut usize,
) -> Result<SingleTableQuery, Error> {
    let mut where_clauses: Vec<String> = Vec::new();
    let mut select_clauses: Vec<String> = Vec::new();

    // Add function_name filter if provided
    if let Some(function_name) = opts.function_name {
        let function_name_param_placeholder = add_parameter(
            function_name,
            ClickhouseType::String,
            query_params,
            param_idx_counter,
        );
        where_clauses.push(format!(
            "i.function_name = {function_name_param_placeholder}"
        ));
    }

    // Add variant_name filter
    if let Some(variant_name) = opts.variant_name {
        let variant_name_param_placeholder = add_parameter(
            variant_name,
            ClickhouseType::String,
            query_params,
            param_idx_counter,
        );
        where_clauses.push(format!("i.variant_name = {variant_name_param_placeholder}"));
    }

    // Add episode_id filter
    if let Some(episode_id) = opts.episode_id {
        let episode_id_param_placeholder = add_parameter(
            episode_id.to_string(),
            ClickhouseType::String,
            query_params,
            param_idx_counter,
        );
        where_clauses.push(format!("i.episode_id = {episode_id_param_placeholder}"));
    }

    // Handle filters
    if let Some(filter_node) = opts.filters {
        let filter_condition_sql = filter_node.to_clickhouse_sql(
            config,
            query_params,
            &mut select_clauses,
            joins,
            param_idx_counter,
        )?;
        where_clauses.push(filter_condition_sql);
    }

    // Add text query filter
    if let Some(search_query_experimental) = opts.search_query_experimental {
        let json_escaped_text_query =
            json_double_escape_string_without_quotes(search_query_experimental)?;
        let text_query_param = add_parameter(
            json_escaped_text_query,
            ClickhouseType::String,
            query_params,
            param_idx_counter,
        );

        // For count, we just need to filter, not return the term frequency
        where_clauses.push(format!(
            "(countSubstringsCaseInsensitiveUTF8(i.input, {text_query_param}) + countSubstringsCaseInsensitiveUTF8(i.output, {text_query_param})) > 0"
        ));
    }

    let select_from_sql_fragment = format!("SELECT toUInt64(COUNT(*)) FROM {table_name} AS i");

    let where_sql_fragment = if where_clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_clauses.join(" AND "))
    };

    Ok(SingleTableQuery {
        select_from_sql_fragment,
        where_sql_fragment,
    })
}

/// Generates the SQL query and parameters for listing inference metadata.
fn generate_list_inference_metadata_sql(
    params: &ListInferenceMetadataParams,
) -> (String, std::collections::HashMap<String, String>) {
    let limit = if params.limit == 0 {
        DEFAULT_INFERENCE_QUERY_LIMIT
    } else {
        params.limit
    };

    let mut where_clauses: Vec<String> = Vec::new();
    let mut query_params: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    // Handle pagination
    let order_direction = match &params.pagination {
        Some(PaginationParams::Before { id }) => {
            query_params.insert("cursor_id".to_string(), id.to_string());
            where_clauses.push("id_uint < toUInt128({cursor_id:UUID})".to_string());
            "DESC"
        }
        Some(PaginationParams::After { id }) => {
            query_params.insert("cursor_id".to_string(), id.to_string());
            where_clauses.push("id_uint > toUInt128({cursor_id:UUID})".to_string());
            "ASC" // For after, we order ASC and then reverse
        }
        None => "DESC", // Default: most recent first
    };

    // Handle function_name filter
    if let Some(function_name) = &params.function_name {
        query_params.insert("function_name".to_string(), function_name.clone());
        where_clauses.push("function_name = {function_name:String}".to_string());
    }

    // Handle variant_name filter
    if let Some(variant_name) = &params.variant_name {
        query_params.insert("variant_name".to_string(), variant_name.clone());
        where_clauses.push("variant_name = {variant_name:String}".to_string());
    }

    // Handle episode_id filter
    if let Some(episode_id) = &params.episode_id {
        query_params.insert("episode_id".to_string(), episode_id.to_string());
        where_clauses.push("episode_id = {episode_id:UUID}".to_string());
    }

    query_params.insert("limit".to_string(), limit.to_string());

    let where_clause = if where_clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_clauses.join(" AND "))
    };

    let query = format!(
        r"
        SELECT
            uint_to_uuid(id_uint) as id,
            function_name,
            variant_name,
            episode_id,
            function_type,
            if(isNull(snapshot_hash), NULL, lower(hex(snapshot_hash))) as snapshot_hash
        FROM InferenceById
        {where_clause}
        ORDER BY id_uint {order_direction}
        LIMIT {{limit:UInt64}}
        FORMAT JSONEachRow
        "
    );

    (query, query_params)
}

/// Generates the ClickHouse query and a list of parameters to be set.
/// The query string will contain placeholders like `{p0:String}`.
/// The returned `Vec<QueryParameter>` contains the mapping from placeholder names (e.g., "p0")
/// to their string values. The client executing the query is responsible for
/// setting these parameters (e.g., via `SET param_p0 = 'value'` or `SET param_p1 = 123`).
///
/// Very important: if a field is missing (fails to join or similar) it will automatically fail the condition.
/// This means that it will not be included in the result set unless the null field is in an OR
/// where another element is true.
///
/// TODOs:
/// - handle selecting the feedback values
pub(crate) fn generate_list_inferences_sql(
    config: &Config,
    opts: &ListInferencesParams<'_>,
) -> Result<(String, Vec<QueryParameter>), Error> {
    opts.validate_pagination()?;

    let mut query_params: Vec<QueryParameter> = Vec::new();
    let mut param_idx_counter = 0;

    let mut sql = match opts.function_name {
        // If function_name is provided, we know which table to query
        // TODO(#4181): list_inferences requests with function name should also query both tables
        // and UNION ALL. The ORDER BY metric join is complicated and net new, so we'll tackle it later.
        Some(function_name) => {
            let mut joins = JoinRegistry::new();
            let function_config = config.get_function(function_name)?;
            let query = generate_single_table_query_for_type(
                config,
                opts,
                function_config.table_name(),
                function_config.config_type() == FunctionConfigType::Chat,
                &mut joins,
                &mut query_params,
                &mut param_idx_counter,
            )?;

            // For single-table queries, use proper ORDER BY generation that supports joining with metrics.
            // Reuse the join registry from the filter so we don't create duplicate joins
            // generate_order_by_sql also always adds id as tie-breaker for deterministic ordering.
            let order_by_sql = generate_order_by_sql(
                opts,
                config,
                &mut query_params,
                &mut param_idx_counter,
                &mut joins,
            )?;

            // Construct final query: SELECT/FROM + JOINs + WHERE + ORDER BY
            let mut sql = query.select_from_sql_fragment;
            if !joins.get_clauses().is_empty() {
                sql.push_str(&joins.get_clauses().join("\n"));
            }
            sql.push('\n');
            sql.push_str(&query.where_sql_fragment);
            if !order_by_sql.is_empty() {
                sql.push_str(&order_by_sql);
            }

            sql
        }
        None => {
            // Otherwise, we need to query both tables with UNION ALL
            let mut chat_joins = JoinRegistry::new();
            let chat_query_snippets = generate_single_table_query_for_type(
                config,
                opts,
                "ChatInference",
                true, // is_chat
                &mut chat_joins,
                &mut query_params,
                &mut param_idx_counter,
            )?;

            let mut json_joins = JoinRegistry::new();
            let json_query_snippets = generate_single_table_query_for_type(
                config,
                opts,
                "JsonInference",
                false, // is_chat
                &mut json_joins,
                &mut query_params,
                &mut param_idx_counter,
            )?;

            // Generate ORDER BY clause for both inner and outer queries
            // For UNION ALL queries, we only support timestamp ordering (not metrics)
            // TODO(#4181): this should support proper ORDER BY generation that supports joining with metrics.
            //
            // For "After" pagination, we need to invert all ordering directions because we'll reverse the list in memory
            let should_invert_directions =
                matches!(opts.pagination, Some(PaginationParams::After { .. }));
            let mut order_clauses = Vec::new();

            // Add user-specified ordering if present
            if let Some(order_by) = opts.order_by {
                for o in order_by {
                    let column = match &o.term {
                        OrderByTerm::Timestamp => "timestamp",
                        OrderByTerm::Metric { name } => {
                            return Err(Error::new(ErrorDetails::InvalidRequest {
                                message: format!(
                                    "ORDER BY metric '{name}' is not supported when querying without function_name"
                                ),
                            }));
                        }
                        OrderByTerm::SearchRelevance => {
                            if opts.search_query_experimental.is_none() {
                                return Err(Error::new(ErrorDetails::InvalidRequest {
                                    message: "ORDER BY relevance requires search_query_experimental in the request".to_string(),
                                }));
                            }
                            "total_term_frequency"
                        }
                    };
                    // Invert direction for "After" pagination since we'll reverse the list
                    let effective_direction = if should_invert_directions {
                        o.direction.inverted()
                    } else {
                        o.direction
                    };
                    let direction = effective_direction.to_clickhouse_direction();
                    order_clauses.push(format!("{column} {direction}"));
                }
            }

            // Always add id as tie-breaker for deterministic ordering
            let id_direction = if let Some(pagination) = &opts.pagination {
                match pagination {
                    PaginationParams::After { .. } => OrderDirection::Asc,
                    PaginationParams::Before { .. } => OrderDirection::Desc,
                }
            } else {
                OrderDirection::Desc
            };

            // For inner queries (chat/json subqueries), use "id" (the actual column name)
            let mut inner_order_by_clauses = order_clauses.clone();
            inner_order_by_clauses.push(format!(
                "toUInt128(id) {}",
                id_direction.to_clickhouse_direction()
            ));

            // Push LIMIT down into each subquery before UNION ALL
            // For UNION ALL, we need to fetch (LIMIT + OFFSET) rows from each table
            // because we don't know which table will contribute to the final result.
            // We then apply the final LIMIT/OFFSET on the outer query.
            //
            // IMPORTANT: When ORDER BY is specified, we add it to each subquery before the LIMIT.
            // Otherwise, the LIMIT will select rows based on the physical table ordering
            // (function_name, variant_name, episode_id), not the user's requested ordering
            // (e.g., timestamp DESC), resulting in incorrect results.
            let inner_limit = opts.limit + opts.offset;

            let inner_limit_param_placeholder = add_parameter(
                inner_limit,
                ClickhouseType::UInt64,
                &mut query_params,
                &mut param_idx_counter,
            );

            // For outer query (after UNION ALL), use "inference_id" (the aliased name)
            let mut outer_order_by_clauses = order_clauses;
            outer_order_by_clauses.push(format!(
                "toUInt128(inference_id) {}",
                id_direction.to_clickhouse_direction()
            ));

            let outer_order_by_sql = format!("\nORDER BY {}", outer_order_by_clauses.join(", "));

            // Combine with UNION ALL and apply outer ORDER BY and LIMIT/OFFSET
            let outer_limit_param_placeholder = add_parameter(
                opts.limit,
                ClickhouseType::UInt64,
                &mut query_params,
                &mut param_idx_counter,
            );

            let outer_offset_clause = if opts.offset != 0 {
                let param_name = add_parameter(
                    opts.offset,
                    ClickhouseType::UInt64,
                    &mut query_params,
                    &mut param_idx_counter,
                );
                format!("OFFSET {param_name}")
            } else {
                String::new()
            };

            // Build inner queries.
            let chat_sql = format!(
                "
            {select_from_clause}
            {joins_clause}
            {where_clause}
            ORDER BY {order_by_clause}
            LIMIT {limit_param}",
                select_from_clause = chat_query_snippets.select_from_sql_fragment,
                joins_clause = chat_joins.get_clauses().join("\n"),
                where_clause = chat_query_snippets.where_sql_fragment,
                order_by_clause = inner_order_by_clauses.join(", "),
                limit_param = inner_limit_param_placeholder
            );
            let json_sql = format!(
                "
            {select_from_clause}
            {joins_clause}
            {where_clause}
            ORDER BY {order_by_clause}
            LIMIT {limit_param}",
                select_from_clause = json_query_snippets.select_from_sql_fragment,
                joins_clause = json_joins.get_clauses().join("\n"),
                where_clause = json_query_snippets.where_sql_fragment,
                order_by_clause = inner_order_by_clauses.join(", "),
                limit_param = inner_limit_param_placeholder
            );

            // Build outer query.
            format!(
                "SELECT * FROM (
                {chat_sql}
                UNION ALL
                {json_sql}
                ) AS combined
                {outer_order_by_sql}
                LIMIT {outer_limit_param_placeholder}
                {outer_offset_clause}"
            )
        }
    };

    // For single-table queries (function_name provided), apply LIMIT/OFFSET at the end
    if opts.function_name.is_some() {
        let limit_param_placeholder = add_parameter(
            opts.limit,
            ClickhouseType::UInt64,
            &mut query_params,
            &mut param_idx_counter,
        );
        sql.push_str(&format!("\nLIMIT {limit_param_placeholder}"));

        if opts.offset != 0 {
            let offset_param_placeholder = add_parameter(
                opts.offset,
                ClickhouseType::UInt64,
                &mut query_params,
                &mut param_idx_counter,
            );
            sql.push_str(&format!("\nOFFSET {offset_param_placeholder}"));
        }
    }

    sql.push_str("\nFORMAT JSONEachRow");

    Ok((sql, query_params))
}

/// Core query building logic for a specific table type
/// Returns a SingleTableQuery with SELECT/FROM and WHERE parts separated,
/// so the caller can insert JOINs between them
fn generate_single_table_query_for_type(
    config: &Config,
    opts: &ListInferencesParams<'_>,
    table_name: &str,
    is_chat: bool,
    joins: &mut JoinRegistry,
    query_params: &mut Vec<QueryParameter>,
    param_idx_counter: &mut usize,
) -> Result<SingleTableQuery, Error> {
    // Use a Vec to maintain explicit column order for UNION ALL compatibility
    // Base columns that are always present, in fixed order
    let mut select_clauses: Vec<String> = vec![];

    // Add type-specific columns first
    if is_chat {
        select_clauses.push("'chat' as type".to_string());
    } else {
        select_clauses.push("'json' as type".to_string());
    }

    // Add remaining columns in alphabetical order
    select_clauses
        .push("formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp".to_string());
    select_clauses.push("i.episode_id as episode_id".to_string());
    select_clauses.push("i.function_name as function_name".to_string());
    select_clauses.push("i.id as inference_id".to_string());
    select_clauses.push("i.input as input".to_string());

    // output will be added later based on output_source
    // Add type-specific columns with consistent names for UNION ALL
    if is_chat {
        select_clauses.push("'' as output_schema".to_string());
    } else {
        select_clauses.push("i.output_schema as output_schema".to_string());
    }

    select_clauses.push("i.tags as tags".to_string());

    if is_chat {
        select_clauses.push("i.tool_params as tool_params".to_string());
        select_clauses.push("i.dynamic_tools as dynamic_tools".to_string());
        select_clauses.push("i.dynamic_provider_tools as dynamic_provider_tools".to_string());
        select_clauses.push("i.allowed_tools as allowed_tools".to_string());
        select_clauses.push("i.tool_choice as tool_choice".to_string());
        select_clauses.push("i.parallel_tool_calls as parallel_tool_calls".to_string());
    } else {
        select_clauses.push("'' as tool_params".to_string());
        select_clauses.push("[] as dynamic_tools".to_string());
        select_clauses.push("[] as dynamic_provider_tools".to_string());
        select_clauses.push("NULL as allowed_tools".to_string());
        select_clauses.push("NULL as tool_choice".to_string());
        select_clauses.push("NULL as parallel_tool_calls".to_string());
    }

    select_clauses.push("i.variant_name as variant_name".to_string());
    select_clauses.push("i.extra_body as extra_body".to_string());
    select_clauses.push("i.inference_params as inference_params".to_string());
    select_clauses.push("i.processing_time_ms as processing_time_ms".to_string());
    select_clauses.push("i.ttft_ms as ttft_ms".to_string());

    let mut where_clauses: Vec<String> = Vec::new();

    // Add function_name filter if provided
    if let Some(function_name) = opts.function_name {
        let function_name_param_placeholder = add_parameter(
            function_name,
            ClickhouseType::String,
            query_params,
            param_idx_counter,
        );
        where_clauses.push(format!(
            "i.function_name = {function_name_param_placeholder}"
        ));
    }

    // Add variant_name filter
    if let Some(variant_name) = opts.variant_name {
        let variant_name_param_placeholder = add_parameter(
            variant_name,
            ClickhouseType::String,
            query_params,
            param_idx_counter,
        );
        where_clauses.push(format!("i.variant_name = {variant_name_param_placeholder}"));
    }

    // Add ids filter
    if let Some(ids) = opts.ids {
        // Our current production_clickhouse_client uses the HTTP client under the hood, which
        // passes parameters in the URL. This will likely hit URL length limits, so instead of passing IDs
        // as a bound parameter, we will write it directly into the query.
        let joined_ids = ids.iter().map(|id| format!("'{id}'")).join(",");
        where_clauses.push(format!("i.id IN [{joined_ids}]"));
    }

    // Add episode_id filter
    if let Some(episode_id) = opts.episode_id {
        let episode_id_param_placeholder = add_parameter(
            episode_id.to_string(),
            ClickhouseType::String,
            query_params,
            param_idx_counter,
        );
        where_clauses.push(format!("i.episode_id = {episode_id_param_placeholder}"));
    }

    // Add before/after pagination filters
    if let Some(ref pagination) = opts.pagination {
        match pagination {
            PaginationParams::Before { id } => {
                let id_param_placeholder = add_parameter(
                    id.to_string(),
                    ClickhouseType::String,
                    query_params,
                    param_idx_counter,
                );
                where_clauses.push(format!(
                    "toUInt128(i.id) < toUInt128(toUUID({id_param_placeholder}))"
                ));
            }
            PaginationParams::After { id } => {
                let id_param_placeholder = add_parameter(
                    id.to_string(),
                    ClickhouseType::String,
                    query_params,
                    param_idx_counter,
                );
                where_clauses.push(format!(
                    "toUInt128(i.id) > toUInt128(toUUID({id_param_placeholder}))"
                ));
            }
        }
    }

    // Handle OutputSource
    match opts.output_source {
        InferenceOutputSource::None | InferenceOutputSource::Inference => {
            // For None, we still select the inference output but it will be dropped
            // when creating the datapoint. This avoids an unnecessary join.
            select_clauses.push("i.output as output".to_string());
        }
        InferenceOutputSource::Demonstration => {
            select_clauses.push("demo_f.value AS output".to_string());
            select_clauses.push("[i.output] as dispreferred_outputs".to_string());

            joins.insert_unchecked(
                "\nJOIN \
                 (SELECT \
                    inference_id, \
                    argMax(value, timestamp) as value \
                  FROM DemonstrationFeedback \
                  GROUP BY inference_id \
                 ) AS demo_f ON i.id = demo_f.inference_id"
                    .to_string(),
            );
        }
    }

    // Handle filters
    if let Some(filter_node) = opts.filters {
        let filter_condition_sql = filter_node.to_clickhouse_sql(
            config,
            query_params,
            &mut select_clauses,
            joins,
            param_idx_counter,
        )?;
        where_clauses.push(filter_condition_sql);
    }

    // Add text query term frequency columns and filter
    if let Some(search_query_experimental) = opts.search_query_experimental {
        let json_escaped_text_query =
            json_double_escape_string_without_quotes(search_query_experimental)?;
        let text_query_param = add_parameter(
            json_escaped_text_query,
            ClickhouseType::String,
            query_params,
            param_idx_counter,
        );

        select_clauses.push(format!(
            "countSubstringsCaseInsensitiveUTF8(i.input, {text_query_param}) as input_term_frequency"
        ));
        select_clauses.push(format!(
            "countSubstringsCaseInsensitiveUTF8(i.output, {text_query_param}) as output_term_frequency"
        ));
        select_clauses.push(
            "input_term_frequency + output_term_frequency as total_term_frequency".to_string(),
        );

        where_clauses.push("total_term_frequency > 0".to_string());
    }

    let select_from_sql_fragment = format!(
        r"SELECT {select_clauses} FROM {table_name} AS i",
        select_clauses = select_clauses.iter().join(",\n    "),
    );

    let where_sql_fragment = if where_clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_clauses.join(" AND "))
    };

    Ok(SingleTableQuery {
        select_from_sql_fragment,
        where_sql_fragment,
    })
}

// ===== Inference count helper functions (merged from inference_count module) =====

/// Builds the SQL query for counting inferences.
fn build_count_inferences_query<'a>(
    params: &'a CountInferencesForFunctionParams<'a>,
) -> (String, HashMap<&'a str, &'a str>) {
    let mut query_params = HashMap::new();
    query_params.insert("function_name", params.function_name);

    let variant_clause = match params.variant_name {
        Some(variant_name) => {
            query_params.insert("variant_name", variant_name);
            "AND variant_name = {variant_name:String}"
        }
        None => "",
    };

    let table_name = params.function_type.table_name();

    let query = format!(
        "SELECT COUNT() AS count
         FROM {table_name}
         WHERE function_name = {{function_name:String}}
           {variant_clause}
         FORMAT JSONEachRow"
    );

    (query, query_params)
}

/// Builds the SQL query for counting inferences grouped by variant.
fn build_count_inferences_by_variant_query<'a>(
    params: &'a CountInferencesForFunctionParams<'a>,
) -> (String, HashMap<&'a str, &'a str>) {
    let mut query_params = HashMap::new();
    query_params.insert("function_name", params.function_name);

    let variant_clause = match params.variant_name {
        Some(variant_name) => {
            query_params.insert("variant_name", variant_name);
            "AND variant_name = {variant_name:String}"
        }
        None => "",
    };

    let table_name = params.function_type.table_name();

    let query = format!(
        "SELECT
            variant_name,
            COUNT() AS inference_count,
            formatDateTime(max(timestamp), '%Y-%m-%dT%H:%i:%S.000Z') AS last_used_at
        FROM {table_name}
        WHERE function_name = {{function_name:String}}
            {variant_clause}
        GROUP BY variant_name
        ORDER BY inference_count DESC
        FORMAT JSONEachRow"
    );

    (query, query_params)
}

/// Build query for counting feedbacks for a boolean/float metric.
/// If `metric_threshold` is Some, filters to only count feedbacks meeting the threshold criteria based on metric type and optimize direction.
fn build_count_metric_feedbacks_query(
    function_name: &str,
    function_type: FunctionConfigType,
    metric_name: &str,
    metric_config: &MetricConfig,
    metric_threshold: Option<f64>,
) -> (String, HashMap<String, String>) {
    let inference_table = function_type.table_name();
    let feedback_table = metric_config.r#type.to_clickhouse_table_name();
    let join_key = metric_config.level.inference_column_name();

    let mut query_params = HashMap::new();

    let value_condition = match metric_threshold {
        None => String::new(),
        Some(threshold) => match metric_config.r#type {
            MetricConfigType::Boolean => match metric_config.optimize {
                MetricConfigOptimize::Max => "AND value = 1",
                MetricConfigOptimize::Min => "AND value = 0",
            }
            .to_string(),
            MetricConfigType::Float => {
                query_params.insert("threshold".to_string(), threshold.to_string());

                let operator = match metric_config.optimize {
                    MetricConfigOptimize::Max => ">",
                    MetricConfigOptimize::Min => "<",
                };
                format!("AND value {operator} {{threshold:Float64}}")
            }
        },
    };

    let query = format!(
        r"SELECT toUInt32(COUNT(*)) as count
        FROM {inference_table} i
        JOIN (
            SELECT target_id, value,
                ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
            FROM {feedback_table}
            WHERE metric_name = {{metric_name:String}}
            {value_condition}
        ) f ON i.{join_key} = f.target_id AND f.rn = 1
        WHERE i.function_name = {{function_name:String}}
        FORMAT JSONEachRow"
    );

    query_params.insert("function_name".to_string(), function_name.to_string());
    query_params.insert("metric_name".to_string(), metric_name.to_string());

    (query, query_params)
}

/// Build query for counting demonstration feedbacks
fn build_count_demonstration_feedbacks_query(
    params: CountInferencesWithDemonstrationFeedbacksParams<'_>,
) -> (String, HashMap<String, String>) {
    let inference_table = params.function_type.table_name();

    let query = format!(
        r"SELECT toUInt32(COUNT(*)) as count
        FROM {inference_table} i
        JOIN (
            SELECT inference_id,
                ROW_NUMBER() OVER (PARTITION BY inference_id ORDER BY timestamp DESC) as rn
            FROM DemonstrationFeedback
        ) f ON i.id = f.inference_id AND f.rn = 1
        WHERE i.function_name = {{function_name:String}}
        FORMAT JSONEachRow"
    );

    let mut query_params = HashMap::new();
    query_params.insert(
        "function_name".to_string(),
        params.function_name.to_string(),
    );

    (query, query_params)
}

/// Converts a time window to a Duration.
fn time_window_to_duration(time_window: &TimeWindow) -> Duration {
    match time_window {
        TimeWindow::Minute => Duration::from_secs(60),
        TimeWindow::Hour => Duration::from_secs(60 * 60),
        TimeWindow::Day => Duration::from_secs(24 * 60 * 60),
        TimeWindow::Week => Duration::from_secs(7 * 24 * 60 * 60),
        TimeWindow::Month => Duration::from_secs(30 * 24 * 60 * 60),
        TimeWindow::Cumulative => Duration::from_secs(365 * 24 * 60 * 60), // 1 year for cumulative
    }
}

/// Build query for getting function throughput by variant
fn build_function_throughput_by_variant_query(
    params: &GetFunctionThroughputByVariantParams<'_>,
) -> (String, HashMap<String, String>) {
    let mut query_params = HashMap::new();
    query_params.insert(
        "function_name".to_string(),
        params.function_name.to_string(),
    );

    let query = match params.time_window {
        TimeWindow::Cumulative => {
            // For cumulative, return all-time data grouped by variant with fixed epoch start
            r"SELECT
                '1970-01-01T00:00:00.000Z' AS period_start,
                i.variant_name AS variant_name,
                toUInt32(count()) AS count
            FROM InferenceById i
            WHERE i.function_name = {function_name:String}
            GROUP BY variant_name
            ORDER BY variant_name DESC
            FORMAT JSONEachRow"
                .to_string()
        }
        TimeWindow::Minute
        | TimeWindow::Hour
        | TimeWindow::Day
        | TimeWindow::Week
        | TimeWindow::Month => {
            // Calculate time delta using idiomatic Duration math in Rust.
            // We use ClickHouse's UUIDv7ToDateTime for timestamp comparison,
            // avoiding manual bit manipulation of UUIDv7 format.
            let time_window_duration = time_window_to_duration(&params.time_window);
            let time_delta = time_window_duration * (params.max_periods + 1);
            let time_delta_secs = time_delta.as_secs();
            query_params.insert("time_delta_secs".to_string(), time_delta_secs.to_string());

            let time_window_str = match params.time_window {
                TimeWindow::Minute => "minute",
                TimeWindow::Hour => "hour",
                TimeWindow::Day => "day",
                TimeWindow::Week => "week",
                TimeWindow::Month => "month",
                TimeWindow::Cumulative => "year", // Won't be reached but makes match exhaustive
            };
            query_params.insert("time_window".to_string(), time_window_str.to_string());

            // Use UUIDv7ToDateTime for timestamp-based filtering.
            // This preserves the original semantics of filtering relative to the max timestamp.
            r"SELECT
                formatDateTime(dateTrunc({time_window:String}, UUIDv7ToDateTime(uint_to_uuid(i.id_uint))), '%Y-%m-%dT%H:%i:%S.000Z') AS period_start,
                i.variant_name AS variant_name,
                toUInt32(count()) AS count
            FROM InferenceById i
            WHERE i.function_name = {function_name:String}
            AND UUIDv7ToDateTime(uint_to_uuid(i.id_uint)) >= (
                SELECT max(UUIDv7ToDateTime(uint_to_uuid(id_uint))) - INTERVAL {time_delta_secs:UInt64} SECOND
                FROM InferenceById
                WHERE function_name = {function_name:String}
            )
            GROUP BY period_start, variant_name
            ORDER BY period_start DESC, variant_name DESC
            FORMAT JSONEachRow".to_string()
        }
    };

    (query, query_params)
}

/// Builds the SQL query for listing functions with inference counts.
fn build_list_functions_with_inference_count_query() -> String {
    r"SELECT
        function_name,
        formatDateTime(max(timestamp), '%Y-%m-%dT%H:%i:%S.000Z') AS last_inference_timestamp,
        toUInt32(count()) AS inference_count
    FROM (
        SELECT function_name, timestamp
        FROM ChatInference
        UNION ALL
        SELECT function_name, timestamp
        FROM JsonInference
    )
    GROUP BY function_name
    ORDER BY last_inference_timestamp DESC
    FORMAT JSONEachRow"
        .to_string()
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use uuid::Uuid;

    use crate::config::{Config, ConfigFileGlob};
    use crate::db::clickhouse::query_builder::test_util::{
        assert_query_contains, assert_query_does_not_contain,
    };
    use crate::db::clickhouse::query_builder::{
        DemonstrationFeedbackFilter, FloatComparisonOperator, FloatMetricFilter, InferenceFilter,
        OrderBy, OrderByTerm, OrderDirection, QueryParameter,
    };
    use crate::db::inferences::{InferenceOutputSource, ListInferencesParams};

    use super::generate_list_inferences_sql;

    async fn get_e2e_config() -> Config {
        // Read the e2e config file
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
    async fn test_query_by_ids_without_function_name_queries_and_unions_both_tables() {
        let config = get_e2e_config().await;
        let id1 = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
        let id2 = Uuid::parse_str("fedcba98-7654-3210-fedc-ba9876543210").unwrap();
        let ids = vec![id1, id2];

        let opts = ListInferencesParams {
            ids: Some(&ids),
            ..Default::default()
        };

        let (sql, _params) = generate_list_inferences_sql(&config, &opts).unwrap();

        // Verify both tables are queried with ORDER BY for deterministic results
        assert_query_contains(
            &sql,
            "SELECT * FROM (
        SELECT
            'chat' as type,
            formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
            i.episode_id as episode_id,
            i.function_name as function_name,
            i.id as inference_id,
            i.input as input,
            '' as output_schema,
            i.tags as tags,
            i.tool_params as tool_params,
            i.dynamic_tools as dynamic_tools,
            i.dynamic_provider_tools as dynamic_provider_tools,
            i.allowed_tools as allowed_tools,
            i.tool_choice as tool_choice,
            i.parallel_tool_calls as parallel_tool_calls,
            i.variant_name as variant_name,
            i.extra_body as extra_body,
            i.inference_params as inference_params,
            i.processing_time_ms as processing_time_ms,
            i.ttft_ms as ttft_ms,
            i.output as output
        FROM
            ChatInference AS i
        WHERE
            i.id IN ['01234567-89ab-cdef-0123-456789abcdef','fedcba98-7654-3210-fedc-ba9876543210']
        ORDER BY toUInt128(id) DESC
        LIMIT {p0:UInt64}
        UNION ALL
        SELECT
            'json' as type,
            formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
            i.episode_id as episode_id,
            i.function_name as function_name,
            i.id as inference_id,
            i.input as input,
            i.output_schema as output_schema,
            i.tags as tags,
            '' as tool_params,
            [] as dynamic_tools,
            [] as dynamic_provider_tools,
            NULL as allowed_tools,
            NULL as tool_choice,
            NULL as parallel_tool_calls,
            i.variant_name as variant_name,
            i.extra_body as extra_body,
            i.inference_params as inference_params,
            i.processing_time_ms as processing_time_ms,
            i.ttft_ms as ttft_ms,
            i.output as output
        FROM
            JsonInference AS i
        WHERE
            i.id IN ['01234567-89ab-cdef-0123-456789abcdef','fedcba98-7654-3210-fedc-ba9876543210']
        ORDER BY toUInt128(id) DESC
        LIMIT {p0:UInt64}
        ) AS combined
        ORDER BY toUInt128(inference_id) DESC
        LIMIT {p1:UInt64}",
        );
    }

    #[tokio::test]
    async fn test_query_by_ids_with_function_name_queries_only_one_table() {
        let config = get_e2e_config().await;
        let id1 = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
        let ids = vec![id1];

        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            ids: Some(&ids),
            ..Default::default()
        };

        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();

        // ORDER BY is always present for deterministic results
        assert_query_contains(
            &sql,
            "SELECT
            'json' as type,
            formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
            i.episode_id as episode_id,
            i.function_name as function_name,
            i.id as inference_id,
            i.input as input,
            i.output_schema as output_schema,
            i.tags as tags,
            '' as tool_params,
            [] as dynamic_tools,
            [] as dynamic_provider_tools,
            NULL as allowed_tools,
            NULL as tool_choice,
            NULL as parallel_tool_calls,
            i.variant_name as variant_name,
            i.extra_body as extra_body,
            i.inference_params as inference_params,
            i.processing_time_ms as processing_time_ms,
            i.ttft_ms as ttft_ms,
            i.output as output
        FROM
            JsonInference AS i
        WHERE
            i.function_name = {p0:String}
            AND i.id IN ['01234567-89ab-cdef-0123-456789abcdef']
        ORDER BY toUInt128(i.id) DESC
        LIMIT {p1:UInt64}",
        );

        // Verify NO UNION ALL
        assert_query_does_not_contain(&sql, "UNION ALL");

        // Verify only JsonInference is queried (extract_entities is a JSON function)
        assert_query_does_not_contain(&sql, "ChatInference");

        assert!(
            params.contains(&QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            }),
            "Function name parameter should be present"
        );
    }

    #[tokio::test]
    async fn test_query_by_ids_with_function_name_and_metric_filter() {
        let config = get_e2e_config().await;
        let id1 = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
        let id2 = Uuid::parse_str("fedcba98-7654-3210-fedc-ba9876543210").unwrap();
        let ids = vec![id1, id2];

        let filter_node = InferenceFilter::FloatMetric(FloatMetricFilter {
            metric_name: "task_success".to_string(),
            comparison_operator: FloatComparisonOperator::GreaterThan,
            value: 0.5,
        });

        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            ids: Some(&ids),
            filters: Some(&filter_node),
            ..Default::default()
        };

        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();

        // Verify NO UNION ALL
        assert_query_does_not_contain(&sql, "UNION ALL");

        // Verify metric filter generates LEFT JOIN
        assert_query_contains(&sql, "
        FROM
            JsonInference AS i
        LEFT JOIN (
            SELECT
                target_id,
                toNullable(argMax(value, timestamp)) as value
            FROM FloatMetricFeedback
            WHERE metric_name = {p1:String}
            GROUP BY target_id
        ) AS j0 ON i.id = j0.target_id
        WHERE
            i.function_name = {p0:String}
            AND i.id IN ['01234567-89ab-cdef-0123-456789abcdef','fedcba98-7654-3210-fedc-ba9876543210']
            AND j0.value > {p2:Float64}");

        assert!(params.contains(&QueryParameter {
            name: "p1".to_string(),
            value: "task_success".to_string(),
        }));
        assert!(params.contains(&QueryParameter {
            name: "p2".to_string(),
            value: "0.5".to_string(),
        }));
    }

    #[tokio::test]
    async fn test_query_with_demonstration_feedback_filter_positive() {
        let config = get_e2e_config().await;

        let filter_node = InferenceFilter::DemonstrationFeedback(DemonstrationFeedbackFilter {
            has_demonstration: true,
        });

        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            filters: Some(&filter_node),
            ..Default::default()
        };

        let (sql, _params) = generate_list_inferences_sql(&config, &opts).unwrap();

        assert_query_contains(
            &sql,
            "i.id IN (SELECT DISTINCT inference_id FROM DemonstrationFeedback)",
        );
    }

    #[tokio::test]
    async fn test_query_with_demonstration_feedback_filter_negative() {
        let config = get_e2e_config().await;

        let filter_node = InferenceFilter::DemonstrationFeedback(DemonstrationFeedbackFilter {
            has_demonstration: false,
        });

        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            filters: Some(&filter_node),
            ..Default::default()
        };

        let (sql, _params) = generate_list_inferences_sql(&config, &opts).unwrap();

        assert_query_contains(
            &sql,
            "i.id NOT IN (SELECT DISTINCT inference_id FROM DemonstrationFeedback)",
        );
    }

    #[tokio::test]
    async fn test_query_by_ids_with_order_by_timestamp() {
        let config = get_e2e_config().await;
        let id1 = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
        let ids = vec![id1];

        let order_by = vec![OrderBy {
            term: OrderByTerm::Timestamp,
            direction: OrderDirection::Desc,
        }];

        let opts = ListInferencesParams {
            ids: Some(&ids),
            order_by: Some(&order_by),
            ..Default::default()
        };

        let (sql, _params) = generate_list_inferences_sql(&config, &opts).unwrap();

        // Verify query is wrapped in subquery with ORDER BY
        assert_query_contains(&sql, "SELECT * FROM");
        assert_query_contains(&sql, ") AS combined");
        assert_query_contains(&sql, "ORDER BY timestamp DESC");

        // Verify ORDER BY appears 3 times: 2 for inner subqueries, 1 for outer query
        // This is critical to ensure correct results when LIMIT is pushed down
        let order_by_count = sql.matches("ORDER BY timestamp DESC").count();
        assert_eq!(
            order_by_count, 3,
            "ORDER BY should appear 3 times: 2 inner + 1 outer to ensure correct LIMIT behavior"
        );

        // Verify LIMIT appears 3 times: 2 for inner subqueries, 1 for outer query
        let limit_count = sql.matches("LIMIT {p").count();
        assert_eq!(
            limit_count, 3,
            "LIMIT should appear 3 times: 2 inner + 1 outer"
        );

        // Verify OFFSET doesn't appear when offset is 0 (default)
        assert_query_does_not_contain(&sql, "OFFSET");
    }

    #[tokio::test]
    async fn test_query_by_ids_without_function_name_returns_errors_for_order_by_metric() {
        let config = get_e2e_config().await;
        let id1 = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
        let ids = vec![id1];

        let order_by = vec![OrderBy {
            term: OrderByTerm::Metric {
                name: "task_success".to_string(),
            },
            direction: OrderDirection::Desc,
        }];

        let opts = ListInferencesParams {
            ids: Some(&ids),
            order_by: Some(&order_by),
            ..Default::default()
        };

        let result = generate_list_inferences_sql(&config, &opts);

        assert!(result.is_err(), "Should return error for ORDER BY metric");

        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("ORDER BY metric"),
            "Error message should indicate metric ordering not supported without function_name"
        );
    }

    #[tokio::test]
    async fn test_query_by_ids_joins_demonstration_output_in_both_subqueries() {
        let config = get_e2e_config().await;
        let id1 = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
        let id2 = Uuid::parse_str("fedcba98-7654-3210-fedc-ba9876543210").unwrap();
        let ids = vec![id1, id2];

        let opts = ListInferencesParams {
            ids: Some(&ids),
            output_source: InferenceOutputSource::Demonstration,
            ..Default::default()
        };

        let (sql, _params) = generate_list_inferences_sql(&config, &opts).unwrap();

        // Verify UNION ALL is present
        assert_query_contains(&sql, "UNION ALL");

        assert_query_contains(
            &sql,
            r"
        FROM ChatInference AS i
        JOIN (SELECT inference_id, argMax(value, timestamp) as value
        FROM DemonstrationFeedback
        GROUP BY inference_id ) AS demo_f ON i.id = demo_f.inference_id",
        );

        assert_query_contains(
            &sql,
            r"
        FROM JsonInference AS i
        JOIN (SELECT inference_id, argMax(value, timestamp) as value
        FROM DemonstrationFeedback
        GROUP BY inference_id ) AS demo_f ON i.id = demo_f.inference_id",
        );
    }

    #[tokio::test]
    async fn test_query_by_ids_without_function_name_with_metric_filter() {
        use crate::db::clickhouse::query_builder::{
            FloatComparisonOperator, FloatMetricFilter, InferenceFilter,
        };

        let config = get_e2e_config().await;
        let id1 = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
        let id2 = Uuid::parse_str("fedcba98-7654-3210-fedc-ba9876543210").unwrap();
        let ids = vec![id1, id2];

        let filter_node = InferenceFilter::FloatMetric(FloatMetricFilter {
            metric_name: "task_success".to_string(),
            comparison_operator: FloatComparisonOperator::GreaterThan,
            value: 0.5,
        });

        let opts = ListInferencesParams {
            ids: Some(&ids),
            filters: Some(&filter_node),
            ..Default::default()
        };

        let (sql, _params) = generate_list_inferences_sql(&config, &opts).unwrap();

        // Verify UNION ALL is present
        assert_query_contains(&sql, "UNION ALL");

        // Verify metric filter join is present
        // The bug would cause this to fail because joins are not inserted
        assert_query_contains(
            &sql,
            r"FROM ChatInference AS i
            LEFT JOIN (
                SELECT
                    target_id,
                    toNullable(argMax(value, timestamp)) as value
                FROM FloatMetricFeedback
                WHERE metric_name = {p0:String}
                GROUP BY target_id
            ) AS j0 ON i.id = j0.target_id",
        );

        assert_query_contains(
            &sql,
            r"FROM JsonInference AS i
            LEFT JOIN (
                SELECT
                    target_id,
                    toNullable(argMax(value, timestamp)) as value
                FROM FloatMetricFeedback
                WHERE metric_name = {p2:String}
                GROUP BY target_id
            ) AS j0 ON i.id = j0.target_id",
        );

        // Verify the metric filter condition is in WHERE
        assert_query_contains(&sql, "j0.value >");
    }

    mod list_inferences_before_after_pagination_tests {
        use super::*;
        use crate::config::Config;
        use crate::db::clickhouse::clickhouse_client::MockClickHouseClient;
        use crate::db::clickhouse::{
            ClickHouseConnectionInfo, ClickHouseResponse, ClickHouseResponseMetadata,
        };
        use crate::db::inferences::{
            InferenceOutputSource, InferenceQueries, ListInferencesParams, PaginationParams,
        };
        use std::sync::Arc;

        #[tokio::test]
        async fn test_list_inferences_with_before() {
            let mut mock_clickhouse_client = MockClickHouseClient::new();
            let before_id = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();

            mock_clickhouse_client
                .expect_run_query_synchronous()
                .withf(move |query, params| {
                    // Should include the before in WHERE clause
                    assert!(
                        query.contains("toUInt128(i.id) < toUInt128(toUUID({p"),
                        "Query should contain before condition"
                    );
                    // Should not have OFFSET when using before/after pagination
                    assert!(
                        !query.contains("OFFSET"),
                        "Query should not have OFFSET with before/after pagination"
                    );
                    // Verify the UUID parameter is set
                    assert!(params.values().any(|v| v.contains("01234567-89ab-cdef")));
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });
            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
            let config = get_e2e_config().await;

            let result = conn
                .list_inferences(
                    &config,
                    &ListInferencesParams {
                        function_name: Some("extract_entities"),
                        pagination: Some(PaginationParams::Before { id: before_id }),
                        output_source: InferenceOutputSource::Inference,
                        limit: 10,
                        offset: 0,
                        ..Default::default()
                    },
                )
                .await
                .unwrap();

            assert_eq!(result.len(), 0);
        }

        #[tokio::test]
        async fn test_list_inferences_with_after() {
            let mut mock_clickhouse_client = MockClickHouseClient::new();
            let after_id = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();

            mock_clickhouse_client
                .expect_run_query_synchronous()
                .withf(move |query, params| {
                    // Should include the after in WHERE clause
                    assert!(
                        query.contains("toUInt128(i.id) > toUInt128(toUUID({p"),
                        "Query should contain after condition"
                    );
                    // Should not have OFFSET when using before/after pagination
                    assert!(
                        !query.contains("OFFSET"),
                        "Query should not have OFFSET with before/after pagination"
                    );
                    // Verify the after UUID parameter is set
                    assert!(params.values().any(|v| v.contains("01234567-89ab-cdef")));
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });
            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
            let config = get_e2e_config().await;

            let result = conn
                .list_inferences(
                    &config,
                    &ListInferencesParams {
                        function_name: Some("extract_entities"),
                        pagination: Some(PaginationParams::After { id: after_id }),
                        output_source: InferenceOutputSource::Inference,
                        limit: 10,
                        offset: 0,
                        ..Default::default()
                    },
                )
                .await
                .unwrap();

            assert_eq!(result.len(), 0);
        }

        #[tokio::test]
        async fn test_list_inferences_before_without_function_name() {
            let mut mock_clickhouse_client = MockClickHouseClient::new();
            let before_id = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();

            mock_clickhouse_client
                .expect_run_query_synchronous()
                .withf(move |query, params| {
                    // Should use UNION ALL query when function_name is not provided
                    assert!(
                        query.contains("UNION ALL"),
                        "Query should use UNION ALL when function_name is not provided"
                    );
                    // Should still include the before in WHERE clause
                    assert!(
                        query.contains("toUInt128(i.id) < toUInt128(toUUID({p"),
                        "Query should contain before condition"
                    );
                    // Verify the before UUID parameter is set
                    assert!(params.values().any(|v| v.contains("01234567-89ab-cdef")));
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });
            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
            let config = Config::default();

            let result = conn
                .list_inferences(
                    &config,
                    &ListInferencesParams {
                        function_name: None, // No function_name to trigger UNION ALL
                        pagination: Some(PaginationParams::Before { id: before_id }),
                        output_source: InferenceOutputSource::Inference,
                        limit: 10,
                        offset: 0,
                        ..Default::default()
                    },
                )
                .await
                .unwrap();

            assert_eq!(result.len(), 0);
        }

        #[tokio::test]
        async fn test_list_inferences_no_before_uses_offset() {
            let mut mock_clickhouse_client = MockClickHouseClient::new();

            mock_clickhouse_client
                .expect_run_query_synchronous()
                .withf(|query, params| {
                    // Should include OFFSET when not using before/after pagination
                    assert!(
                        query.contains("OFFSET {p"),
                        "Query should have OFFSET without before/after pagination"
                    );
                    // Should have offset parameter set to 20
                    assert!(params.values().any(|v| *v == "20"));
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });
            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
            let config = get_e2e_config().await;

            let result = conn
                .list_inferences(
                    &config,
                    &ListInferencesParams {
                        function_name: Some("extract_entities"),
                        pagination: None, // No before/after pagination
                        output_source: InferenceOutputSource::Inference,
                        limit: 10,
                        offset: 20, // Should use offset
                        ..Default::default()
                    },
                )
                .await
                .unwrap();

            assert_eq!(result.len(), 0);
        }

        #[tokio::test]
        async fn test_list_inferences_before_with_timestamp_ordering_succeeds() {
            let config = get_e2e_config().await;
            let before_id = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();

            // Timestamp ordering should work with before/after pagination
            let result = generate_list_inferences_sql(
                &config,
                &ListInferencesParams {
                    function_name: Some("extract_entities"),
                    pagination: Some(PaginationParams::Before { id: before_id }),
                    output_source: InferenceOutputSource::Inference,
                    limit: 10,
                    offset: 0,
                    order_by: Some(&[OrderBy {
                        term: OrderByTerm::Timestamp,
                        direction: OrderDirection::Desc,
                    }]),
                    ..Default::default()
                },
            );

            assert!(
                result.is_ok(),
                "Timestamp ordering should work with before/after pagination"
            );
            let (sql, _) = result.unwrap();
            // Should include timestamp ordering
            assert!(
                sql.contains("i.timestamp DESC"),
                "SQL should include timestamp ordering"
            );
            // Should include id as tie-breaker
            assert!(
                sql.contains("toUInt128(i.id) DESC"),
                "SQL should include id as tie-breaker"
            );
        }

        #[tokio::test]
        async fn test_list_inferences_before_with_metric_ordering_fails() {
            let config = get_e2e_config().await;
            let before_id = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();

            // Metric ordering should NOT work with before/after pagination
            let result = generate_list_inferences_sql(
                &config,
                &ListInferencesParams {
                    function_name: Some("extract_entities"),
                    pagination: Some(PaginationParams::Before { id: before_id }),
                    output_source: InferenceOutputSource::Inference,
                    limit: 10,
                    offset: 0,
                    order_by: Some(&[OrderBy {
                        term: OrderByTerm::Metric {
                            name: "test_metric".to_string(),
                        },
                        direction: OrderDirection::Desc,
                    }]),
                    ..Default::default()
                },
            );

            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                err.to_string().contains(
                    "only ordering by timestamp is supported with before/after pagination"
                ),
                "Error should mention metric ordering conflict with before/after pagination"
            );
        }

        #[tokio::test]
        async fn test_list_inferences_before_with_search_relevance_fails() {
            let config = get_e2e_config().await;
            let before_id = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();

            // Search relevance ordering should NOT work with before/after pagination
            let result = generate_list_inferences_sql(
                &config,
                &ListInferencesParams {
                    function_name: Some("extract_entities"),
                    pagination: Some(PaginationParams::Before { id: before_id }),
                    output_source: InferenceOutputSource::Inference,
                    limit: 10,
                    offset: 0,
                    order_by: Some(&[OrderBy {
                        term: OrderByTerm::SearchRelevance,
                        direction: OrderDirection::Desc,
                    }]),
                    ..Default::default()
                },
            );

            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                err.to_string().contains(
                    "only ordering by timestamp is supported with before/after pagination"
                ),
                "Error should mention search relevance ordering conflict with before/after pagination"
            );
        }
    }

    mod get_inference_output_tests {
        use crate::db::clickhouse::clickhouse_client::MockClickHouseClient;
        use crate::db::clickhouse::query_builder::test_util::assert_query_contains;
        use crate::db::clickhouse::{
            ClickHouseConnectionInfo, ClickHouseResponse, ClickHouseResponseMetadata,
        };
        use crate::db::inferences::{FunctionInfo, InferenceQueries};
        use crate::inference::types::FunctionType;
        use std::sync::Arc;
        use uuid::Uuid;

        #[tokio::test]
        async fn test_get_inference_output_chat_inference_success() {
            let inference_id = Uuid::now_v7();
            let episode_id = Uuid::now_v7();
            let function_info = FunctionInfo {
                function_name: "test_chat_function".to_string(),
                function_type: FunctionType::Chat,
                variant_name: "test_variant".to_string(),
                episode_id,
            };

            let mut mock = MockClickHouseClient::new();
            mock.expect_run_query_synchronous()
                .withf(move |query, params| {
                    // Verify query targets ChatInference table
                    assert_query_contains(query, "FROM ChatInference");
                    // Verify parameterized WHERE clause
                    assert_query_contains(query, "id = {inference_id:String}");
                    assert_query_contains(query, "episode_id = {episode_id:UUID}");
                    assert_query_contains(query, "function_name = {function_name:String}");
                    assert_query_contains(query, "variant_name = {variant_name:String}");
                    // Verify parameters are set
                    assert_eq!(
                        params.get("function_name"),
                        Some(&"test_chat_function"),
                        "function_name parameter should be set"
                    );
                    assert_eq!(
                        params.get("variant_name"),
                        Some(&"test_variant"),
                        "variant_name parameter should be set"
                    );
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: r#"{"output":"[{\"type\":\"text\",\"text\":\"Hello!\"}]"}"#
                            .to_string(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 1,
                            written_rows: 0,
                        },
                    })
                });

            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
            let result = conn
                .get_inference_output(&function_info, inference_id)
                .await
                .expect("Should succeed");

            assert!(
                result.is_some(),
                "Should return Some for existing inference"
            );
            assert_eq!(
                result.unwrap(),
                r#"[{"type":"text","text":"Hello!"}]"#,
                "Should return the output string"
            );
        }

        #[tokio::test]
        async fn test_get_inference_output_json_inference_success() {
            let inference_id = Uuid::now_v7();
            let episode_id = Uuid::now_v7();
            let function_info = FunctionInfo {
                function_name: "test_json_function".to_string(),
                function_type: FunctionType::Json,
                variant_name: "json_variant".to_string(),
                episode_id,
            };

            let mut mock = MockClickHouseClient::new();
            mock.expect_run_query_synchronous()
                .withf(move |query, params| {
                    // Verify query targets JsonInference table
                    assert_query_contains(query, "FROM JsonInference");
                    // Verify parameters
                    assert_eq!(
                        params.get("function_name"),
                        Some(&"test_json_function"),
                        "function_name parameter should be set"
                    );
                    assert_eq!(
                        params.get("variant_name"),
                        Some(&"json_variant"),
                        "variant_name parameter should be set"
                    );
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: r#"{"output":"{\"raw\":\"{\\\"score\\\":0.95}\",\"parsed\":{\"score\":0.95}}"}"#
                            .to_string(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 1,
                            written_rows: 0,
                        },
                    })
                });

            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
            let result = conn
                .get_inference_output(&function_info, inference_id)
                .await
                .expect("Should succeed");

            assert!(
                result.is_some(),
                "Should return Some for existing JSON inference"
            );
            assert!(
                result.unwrap().contains("score"),
                "Should return the JSON output"
            );
        }

        #[tokio::test]
        async fn test_get_inference_output_not_found() {
            let inference_id = Uuid::now_v7();
            let episode_id = Uuid::now_v7();
            let function_info = FunctionInfo {
                function_name: "nonexistent_function".to_string(),
                function_type: FunctionType::Chat,
                variant_name: "nonexistent_variant".to_string(),
                episode_id,
            };

            let mut mock = MockClickHouseClient::new();
            mock.expect_run_query_synchronous().returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(), // Empty response indicates not found
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
            let result = conn
                .get_inference_output(&function_info, inference_id)
                .await
                .expect("Should succeed even when not found");

            assert!(
                result.is_none(),
                "Should return None for non-existent inference"
            );
        }

        #[tokio::test]
        async fn test_get_inference_output_uses_all_parameters_for_security() {
            let inference_id = Uuid::now_v7();
            let episode_id = Uuid::now_v7();
            let function_info = FunctionInfo {
                function_name: "secure_function".to_string(),
                function_type: FunctionType::Chat,
                variant_name: "secure_variant".to_string(),
                episode_id,
            };

            let mut mock = MockClickHouseClient::new();
            mock.expect_run_query_synchronous()
                .withf(move |query, params| {
                    // Verify ALL parameters are used in WHERE clause for security
                    // This prevents unauthorized access by requiring all identifiers to match
                    assert_query_contains(query, "id = {inference_id:String}");
                    assert_query_contains(query, "episode_id = {episode_id:UUID}");
                    assert_query_contains(query, "function_name = {function_name:String}");
                    assert_query_contains(query, "variant_name = {variant_name:String}");

                    // Verify all parameters are bound (not interpolated)
                    assert!(
                        params.contains_key("inference_id"),
                        "inference_id should be a bound parameter"
                    );
                    assert!(
                        params.contains_key("episode_id"),
                        "episode_id should be a bound parameter"
                    );
                    assert!(
                        params.contains_key("function_name"),
                        "function_name should be a bound parameter"
                    );
                    assert!(
                        params.contains_key("variant_name"),
                        "variant_name should be a bound parameter"
                    );

                    // Verify no string interpolation in the query (SQL injection prevention)
                    assert!(
                        !query.contains("secure_function"),
                        "function_name should NOT be interpolated into query"
                    );
                    assert!(
                        !query.contains("secure_variant"),
                        "variant_name should NOT be interpolated into query"
                    );
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: r#"{"output":"test output"}"#.to_string(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 1,
                            written_rows: 0,
                        },
                    })
                });

            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
            let result = conn
                .get_inference_output(&function_info, inference_id)
                .await;

            assert!(result.is_ok(), "Query should execute successfully");
        }

        #[tokio::test]
        async fn test_get_inference_output_table_selection_by_function_type() {
            // Test Chat function type uses ChatInference table
            let chat_function_info = FunctionInfo {
                function_name: "chat_func".to_string(),
                function_type: FunctionType::Chat,
                variant_name: "v1".to_string(),
                episode_id: Uuid::now_v7(),
            };

            let mut chat_mock = MockClickHouseClient::new();
            chat_mock
                .expect_run_query_synchronous()
                .withf(|query, _| {
                    assert_query_contains(query, "FROM ChatInference");
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });

            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(chat_mock));
            let _ = conn
                .get_inference_output(&chat_function_info, Uuid::now_v7())
                .await;

            // Test Json function type uses JsonInference table
            let json_function_info = FunctionInfo {
                function_name: "json_func".to_string(),
                function_type: FunctionType::Json,
                variant_name: "v1".to_string(),
                episode_id: Uuid::now_v7(),
            };

            let mut json_mock = MockClickHouseClient::new();
            json_mock
                .expect_run_query_synchronous()
                .withf(|query, _| {
                    assert_query_contains(query, "FROM JsonInference");
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });

            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(json_mock));
            let _ = conn
                .get_inference_output(&json_function_info, Uuid::now_v7())
                .await;
        }
    }

    mod list_inference_metadata_tests {
        use crate::db::clickhouse::clickhouse_client::MockClickHouseClient;
        use crate::db::clickhouse::query_builder::test_util::{
            assert_query_contains, assert_query_does_not_contain,
        };
        use crate::db::clickhouse::{
            ClickHouseConnectionInfo, ClickHouseResponse, ClickHouseResponseMetadata,
        };
        use crate::db::inferences::{
            InferenceQueries, ListInferenceMetadataParams, PaginationParams,
        };
        use std::sync::Arc;
        use uuid::Uuid;

        #[tokio::test]
        async fn test_list_inference_metadata_no_pagination() {
            let mut mock = MockClickHouseClient::new();
            mock.expect_run_query_synchronous()
                .withf(|query, _params| {
                    assert_query_contains(query, "uint_to_uuid(id_uint) as id");
                    assert_query_contains(query, "FROM InferenceById");
                    assert_query_contains(query, "ORDER BY id_uint DESC");
                    assert_query_does_not_contain(query, "WHERE");
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });

            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
            let result = conn
                .list_inference_metadata(&ListInferenceMetadataParams::default())
                .await
                .unwrap();

            assert!(result.is_empty());
        }

        #[tokio::test]
        async fn test_list_inference_metadata_with_before() {
            let cursor_id = Uuid::now_v7();
            let mut mock = MockClickHouseClient::new();
            mock.expect_run_query_synchronous()
                .withf(move |query, params| {
                    assert_query_contains(query, "id_uint < toUInt128({cursor_id:UUID})");
                    assert_query_contains(query, "ORDER BY id_uint DESC");
                    assert!(params.get("cursor_id").is_some());
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });

            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
            let result = conn
                .list_inference_metadata(&ListInferenceMetadataParams {
                    pagination: Some(PaginationParams::Before { id: cursor_id }),
                    limit: 10,
                    ..Default::default()
                })
                .await
                .unwrap();

            assert!(result.is_empty());
        }

        #[tokio::test]
        async fn test_list_inference_metadata_with_after() {
            let cursor_id = Uuid::now_v7();
            let mut mock = MockClickHouseClient::new();
            mock.expect_run_query_synchronous()
                .withf(move |query, params| {
                    assert_query_contains(query, "id_uint > toUInt128({cursor_id:UUID})");
                    assert_query_contains(query, "ORDER BY id_uint ASC");
                    assert!(params.get("cursor_id").is_some());
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });

            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
            let result = conn
                .list_inference_metadata(&ListInferenceMetadataParams {
                    pagination: Some(PaginationParams::After { id: cursor_id }),
                    limit: 10,
                    ..Default::default()
                })
                .await
                .unwrap();

            assert!(result.is_empty());
        }

        #[tokio::test]
        async fn test_list_inference_metadata_with_function_name() {
            let mut mock = MockClickHouseClient::new();
            mock.expect_run_query_synchronous()
                .withf(|query, params| {
                    assert_query_contains(query, "function_name = {function_name:String}");
                    assert_query_contains(query, "WHERE");
                    assert_eq!(params.get("function_name"), Some(&"test_function"));
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });

            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
            let result = conn
                .list_inference_metadata(&ListInferenceMetadataParams {
                    function_name: Some("test_function".to_string()),
                    ..Default::default()
                })
                .await
                .unwrap();

            assert!(result.is_empty());
        }

        #[tokio::test]
        async fn test_list_inference_metadata_with_variant_name() {
            let mut mock = MockClickHouseClient::new();
            mock.expect_run_query_synchronous()
                .withf(|query, params| {
                    assert_query_contains(query, "variant_name = {variant_name:String}");
                    assert_query_contains(query, "WHERE");
                    assert_eq!(params.get("variant_name"), Some(&"test_variant"));
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });

            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
            let result = conn
                .list_inference_metadata(&ListInferenceMetadataParams {
                    variant_name: Some("test_variant".to_string()),
                    ..Default::default()
                })
                .await
                .unwrap();

            assert!(result.is_empty());
        }

        #[tokio::test]
        async fn test_list_inference_metadata_with_episode_id() {
            let episode_id = Uuid::now_v7();
            let mut mock = MockClickHouseClient::new();
            mock.expect_run_query_synchronous()
                .withf(move |query, params| {
                    assert_query_contains(query, "episode_id = {episode_id:UUID}");
                    assert_query_contains(query, "WHERE");
                    assert!(params.get("episode_id").is_some());
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });

            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
            let result = conn
                .list_inference_metadata(&ListInferenceMetadataParams {
                    episode_id: Some(episode_id),
                    ..Default::default()
                })
                .await
                .unwrap();

            assert!(result.is_empty());
        }

        #[tokio::test]
        async fn test_list_inference_metadata_with_all_filters() {
            let episode_id = Uuid::now_v7();
            let mut mock = MockClickHouseClient::new();
            mock.expect_run_query_synchronous()
                .withf(move |query, params| {
                    assert_query_contains(query, "function_name = {function_name:String}");
                    assert_query_contains(query, "variant_name = {variant_name:String}");
                    assert_query_contains(query, "episode_id = {episode_id:UUID}");
                    assert_query_contains(query, "WHERE");
                    assert_eq!(params.get("function_name"), Some(&"test_function"));
                    assert_eq!(params.get("variant_name"), Some(&"test_variant"));
                    assert!(params.get("episode_id").is_some());
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });

            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
            let result = conn
                .list_inference_metadata(&ListInferenceMetadataParams {
                    function_name: Some("test_function".to_string()),
                    variant_name: Some("test_variant".to_string()),
                    episode_id: Some(episode_id),
                    ..Default::default()
                })
                .await
                .unwrap();

            assert!(result.is_empty());
        }
    }
}
