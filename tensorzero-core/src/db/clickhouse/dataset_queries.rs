use async_trait::async_trait;
use itertools::Itertools;
use std::collections::HashMap;
use std::num::ParseIntError;
use tokio::try_join;
use uuid::Uuid;

use crate::db::clickhouse::query_builder::QueryParameter;
use crate::db::clickhouse::{
    escape_string_for_clickhouse_literal, ClickHouseConnectionInfo, ExternalDataInfo,
};
use crate::endpoints::datasets::v1::types::{DatapointOrderBy, DatapointOrderByTerm};
use crate::endpoints::shared_types::OrderDirection;
// TODO: move things somewhere sensible
use crate::db::datasets::{
    ChatInferenceDatapointInsert, CountDatapointsForDatasetFunctionParams, DatapointInsert,
    DatasetMetadata, DatasetOutputSource, DatasetQueries, DatasetQueryParams, GetDatapointParams,
    GetDatapointsParams, GetDatasetMetadataParams, JsonInferenceDatapointInsert,
};
use crate::endpoints::datasets::{validate_dataset_name, DatapointKind, StoredDatapoint};
use crate::error::{Error, ErrorDetails};

#[async_trait]
impl DatasetQueries for ClickHouseConnectionInfo {
    async fn count_rows_for_dataset(&self, params: &DatasetQueryParams) -> Result<u32, Error> {
        // Validate that no limit or offset is provided
        if params.limit.is_some() || params.offset.is_some() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "limit and offset are not supported for count_rows_for_dataset"
                    .to_string(),
            }));
        }

        let (query, query_params_owned) =
            build_select_inferences_matching_dataset_subquery(params)?;

        let query_params: HashMap<_, _> = query_params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let count_query = format!("SELECT toUInt32(count()) as count FROM ({query})");
        let response = self
            .run_query_synchronous(count_query, &query_params)
            .await?;

        let count_str = response.response.trim();
        let count: u32 = count_str.parse().map_err(|e: ParseIntError| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: e.to_string(),
            })
        })?;
        Ok(count)
    }

    async fn insert_rows_for_dataset(&self, params: &DatasetQueryParams) -> Result<u32, Error> {
        // Validate that dataset_name is provided
        let dataset_name = params.dataset_name.as_ref().ok_or_else(|| {
            Error::new(ErrorDetails::InvalidRequest {
                message: "dataset_name is required for dataset insertion".to_string(),
            })
        })?;
        validate_dataset_name(dataset_name)?;

        // Determine the destination table based on the inference type
        let destination_table = params.inference_type.table_name().as_str();

        // Build the SELECT query from the source table
        // TODO: use query_builder module for this in the future.
        let (source_query, mut query_params_owned) =
            build_select_inferences_matching_dataset_subquery(params)?;

        // Add additional query parameters
        query_params_owned.insert("datapoint_table".to_string(), destination_table.to_string());
        query_params_owned.insert("dataset_name".to_string(), dataset_name.clone());

        // Build the INSERT query with conditional logic based on inference type
        let type_specific_fields = match params.inference_type {
            DatapointKind::Chat => {
                r"subquery.tool_params,
    subquery.dynamic_tools,
    subquery.dynamic_provider_tools,
    subquery.parallel_tool_calls,
    subquery.tool_choice,
    subquery.allowed_tools,
            "
            }
            DatapointKind::Json => "subquery.output_schema,",
        };

        let wrapped_query = format!(
            r"
            INSERT INTO {{datapoint_table:Identifier}}
            SELECT
                {{dataset_name:String}} as dataset_name,
                subquery.function_name as function_name,
                generateUUIDv7() as id,
                subquery.episode_id as episode_id,
                subquery.input as input,
                subquery.output as output,
                {type_specific_fields}
                subquery.tags as tags,
                subquery.auxiliary as auxiliary,
                false as is_deleted,
                now64() as updated_at,
                null as staled_at,
                subquery.id as source_inference_id,
                false as is_custom,
                subquery.name as name
            FROM (
                {source_query}
            ) AS subquery
            LEFT JOIN {{datapoint_table:Identifier}} as existing FINAL
                ON {{dataset_name:String}} = existing.dataset_name
                   AND subquery.function_name = existing.function_name
                   AND subquery.id = existing.source_inference_id
                   AND existing.staled_at IS NULL
            WHERE existing.source_inference_id IS NULL
            "
        );

        // Convert query params to the format needed by run_query_synchronous
        let query_params: HashMap<_, _> = query_params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        // Execute the INSERT query
        let response = self
            .run_query_synchronous(wrapped_query, &query_params)
            .await?;

        // Parse the response to get the number of rows inserted
        // ClickHouse returns summary information in the metadata
        Ok(response.metadata.written_rows as u32)
    }

    async fn get_dataset_metadata(
        &self,
        params: &GetDatasetMetadataParams,
    ) -> Result<Vec<DatasetMetadata>, Error> {
        let mut query_params_owned = HashMap::<String, String>::new();

        // Build clauses and query parameters
        let function_where_clause = match &params.function_name {
            Some(fn_name) => {
                query_params_owned.insert("function_name".to_string(), fn_name.clone());
                "AND function_name = {function_name:String}"
            }
            None => "",
        };
        let limit_clause = match params.limit {
            Some(limit) => {
                query_params_owned.insert("limit".to_string(), limit.to_string());
                "LIMIT {limit:UInt32}"
            }
            None => "",
        };

        let offset_clause = match params.offset {
            Some(offset) => {
                query_params_owned.insert("offset".to_string(), offset.to_string());
                "OFFSET {offset:UInt32}"
            }
            None => "",
        };

        let query = format!(
            r"
            SELECT
                dataset_name,
                toUInt32(sum(count)) AS count,
                formatDateTime(max(last_updated), '%Y-%m-%dT%H:%i:%SZ') AS last_updated
            FROM (
                SELECT
                    dataset_name,
                    toUInt32(count()) AS count,
                    max(updated_at) AS last_updated
                FROM ChatInferenceDatapoint
                FINAL
                WHERE staled_at IS NULL
                {function_where_clause}
                GROUP BY dataset_name
                UNION ALL
                SELECT
                    dataset_name,
                    toUInt32(count()) AS count,
                    max(updated_at) AS last_updated
                FROM JsonInferenceDatapoint
                FINAL
                WHERE staled_at IS NULL
                {function_where_clause}
                GROUP BY dataset_name
            )
            GROUP BY dataset_name
            ORDER BY last_updated DESC
            {limit_clause}
            {offset_clause}
            FORMAT JSONEachRow
            "
        );

        let query_params = query_params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let response = self.run_query_synchronous(query, &query_params).await?;

        // Parse the response as JSON lines
        response
            .response
            .trim()
            .lines()
            .map(|row| {
                serde_json::from_str(row).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: format!("Failed to deserialize DatasetMetadata: {e}"),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()
    }

    async fn count_datasets(&self) -> Result<u32, Error> {
        let query = r"
            SELECT
                toUInt32(uniqExact(dataset_name)) as count
            FROM (
                SELECT dataset_name
                FROM ChatInferenceDatapoint FINAL
                WHERE staled_at IS NULL
                UNION ALL
                SELECT dataset_name
                FROM JsonInferenceDatapoint FINAL
                WHERE staled_at IS NULL
            )";

        let response = self
            .run_query_synchronous(query.to_string(), &HashMap::new())
            .await?;

        let count_str = response.response.trim();
        let count: u32 = count_str.parse().map_err(|e: ParseIntError| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: e.to_string(),
            })
        })?;
        Ok(count)
    }

    async fn count_datapoints_for_dataset_function(
        &self,
        params: &CountDatapointsForDatasetFunctionParams,
    ) -> Result<u32, Error> {
        let query = "
        SELECT toUInt32(count()) as count
        FROM {table:Identifier}
        WHERE dataset_name = {dataset_name:String}
            AND function_name = {function_name:String}";

        let mut query_params = HashMap::new();
        let table_name = params.function_type.table_name();
        query_params.insert("table", table_name.as_str());
        query_params.insert("dataset_name", params.dataset_name.as_str());
        query_params.insert("function_name", params.function_name.as_str());

        let response = self
            .run_query_synchronous(query.to_string(), &query_params)
            .await?;

        let count_str = response.response.trim();
        let count: u32 = count_str.parse().map_err(|e: ParseIntError| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: e.to_string(),
            })
        })?;
        Ok(count)
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

        // TODO(shuyangli): Consider checking if multiple datapoints came back.
        let first_datapoint = datapoints.swap_remove(0);
        Ok(first_datapoint)
    }

    async fn get_datapoints(
        &self,
        params: &GetDatapointsParams,
    ) -> Result<Vec<StoredDatapoint>, Error> {
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
        let limit_str = limit.to_string();
        let offset_str = offset.to_string();
        let subquery_limit_str = (limit + offset).to_string();

        // If neither IDs nor dataset are provided, reject the query.
        if dataset_name.is_none() && ids.is_none() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "At least one of dataset_name or ids must be provided".to_string(),
            }));
        }

        // If IDs are provided, they must not be empty.
        if let Some(ids_vec) = ids {
            if ids_vec.is_empty() {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "ids must not be an empty list".to_string(),
                }));
            }
        }

        // Build params and where clauses.
        let mut query_params = HashMap::new();
        query_params.insert("limit", limit_str.as_str());
        query_params.insert("offset", offset_str.as_str());
        query_params.insert("subquery_limit", subquery_limit_str.as_str());

        let dataset_name_clause = match dataset_name {
            Some(dataset_name) => {
                query_params.insert("dataset_name", dataset_name.as_str());
                "AND dataset_name = {dataset_name:String}"
            }
            None => "",
        };

        let function_name_clause = match function_name {
            Some(function_name) => {
                query_params.insert("function_name", function_name.as_str());
                "AND function_name = {function_name:String}"
            }
            None => "",
        };

        let ids_clause = match ids {
            None => String::new(),
            Some(ids_vec) => {
                // Our current production_clickhouse_client uses the HTTP client under the hood, which
                // passes parameters in the URL. This will likely hit URL length limits, so instead of passing IDs
                // as a bound parameter, we will write it directly into the query.
                let joined_ids = ids_vec.iter().map(|id| format!("'{id}'")).join(",");
                format!("AND id IN [{joined_ids}]")
            }
        };

        let allow_stale_clause = if *allow_stale {
            ""
        } else {
            "AND staled_at IS NULL"
        };

        // Generate filter SQL clause if provided
        let (filter_clause, filter_params) = if let Some(filter) = filter {
            let (filter_sql, filter_params) = filter.to_clickhouse_sql("i");
            (format!("AND {filter_sql}"), filter_params)
        } else {
            (String::new(), Vec::new())
        };

        // Add filter parameters to query_params.
        for QueryParameter { name, value } in &filter_params {
            query_params.insert(name.as_str(), value.as_str());
        }

        // Add text query term frequency columns and filter
        let (search_select_clauses, search_filter_clause) = if let Some(search_query) =
            search_query_experimental
        {
            // JSON-escape the query for matching against JSON-serialized input/output
            let escaped_query = json_escape_string_without_quotes(search_query)?;
            let clickhouse_escaped = escape_string_for_clickhouse_literal(&escaped_query);

            let select_clauses = format!(
                r"ifNull(countSubstringsCaseInsensitiveUTF8(input, '{clickhouse_escaped}'), 0) as input_term_frequency,
                ifNull(countSubstringsCaseInsensitiveUTF8(output, '{clickhouse_escaped}'), 0) as output_term_frequency,
                input_term_frequency + output_term_frequency as total_term_frequency"
            );
            let filter_clause = "AND total_term_frequency > 0";
            (format!(", {select_clauses}"), filter_clause)
        } else {
            (String::new(), "")
        };

        // Generate ORDER BY clause
        let order_by_clause =
            get_order_by_clause(order_by.as_ref(), search_query_experimental.as_ref())?;

        // When constructing the query, all filters are pushed down to the subqueries for Chat/Json table, and the final
        // SELECT only handles merging and ordering for pagination.
        let query = format!(
            r"
        WITH dataset as (
            SELECT
                'chat' as type,
                dataset_name,
                function_name,
                name,
                id,
                episode_id,
                input,
                output,
                tool_params,
                dynamic_tools,
                dynamic_provider_tools,
                parallel_tool_calls,
                tool_choice,
                allowed_tools,
                '' as output_schema, -- for column alignment in UNION ALL
                tags,
                auxiliary,
                source_inference_id,
                is_deleted,
                is_custom,
                staled_at,
                formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at
                {search_select_clauses}
            FROM ChatInferenceDatapoint AS i FINAL
            WHERE true
                {dataset_name_clause}
                {function_name_clause}
                {ids_clause}
                {allow_stale_clause}
                {filter_clause}
                {search_filter_clause}
            {order_by_clause}
            LIMIT {{subquery_limit:UInt32}}
            UNION ALL
            SELECT
                'json' as type,
                dataset_name,
                function_name,
                name,
                id,
                episode_id,
                input,
                output,
                '' as tool_params, -- for column alignment in UNION ALL
                [] as dynamic_tools,
                [] as dynamic_provider_tools,
                NULL as parallel_tool_calls,
                NULL as tool_choice,
                NULL as allowed_tools,
                output_schema,
                tags,
                auxiliary,
                source_inference_id,
                is_deleted,
                is_custom,
                staled_at,
                formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at
                {search_select_clauses}
            FROM JsonInferenceDatapoint AS i FINAL
            WHERE true
                {dataset_name_clause}
                {function_name_clause}
                {ids_clause}
                {allow_stale_clause}
                {filter_clause}
                {search_filter_clause}
            {order_by_clause}
            LIMIT {{subquery_limit:UInt32}}
        )
        SELECT * FROM dataset
        {order_by_clause}
        LIMIT {{limit:UInt32}}
        OFFSET {{offset:UInt32}}
        FORMAT JSONEachRow
        "
        );

        let result = self
            .run_query_synchronous(query.to_string(), &query_params)
            .await?;

        if result.response.is_empty() {
            return Ok(Vec::new());
        }

        // Parse each line as a separate datapoint
        let mut datapoints = Vec::with_capacity(params.ids.as_ref().map_or(0, Vec::len));
        for line in result.response.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let datapoint: StoredDatapoint = serde_json::from_str(line).map_err(|e| {
                Error::new(ErrorDetails::ClickHouseDeserialization {
                    message: format!("Failed to deserialize datapoint: {e}"),
                })
            })?;
            datapoints.push(datapoint);
        }

        Ok(datapoints)
    }

    /// Deletes datapoints from a dataset.
    /// If datapoint_ids is empty, all datapoints in the dataset will be deleted.
    /// Otherwise, only the datapoints with the given IDs will be deleted.
    ///
    /// Returns the number of datapoints that were deleted.
    async fn delete_datapoints(
        &self,
        dataset_name: &str,
        datapoint_ids: Option<&[Uuid]>,
    ) -> Result<u64, Error> {
        let datapoint_ids_filter_clause = match datapoint_ids {
            None => Ok(String::new()),
            Some(datapoint_ids) => {
                if datapoint_ids.is_empty() {
                    Err(Error::new(ErrorDetails::InvalidRequest {
                        message: "If datapoint_ids are provided as a vector, it must be non-empty"
                            .to_string(),
                    }))
                } else {
                    Ok(format!(
                        "AND id IN [{}]",
                        datapoint_ids.iter().map(|id| format!("'{id}'")).join(",")
                    ))
                }
            }
        }?;

        // NOTE: in the two queries below, we don't alias to staled_at because then we won't select any rows.
        let chat_query = format!(
            r"
            WITH existing_chat_datapoints AS (
                SELECT *
                FROM ChatInferenceDatapoint FINAL
                WHERE dataset_name = {{dataset_name:String}}
                {datapoint_ids_filter_clause}
                AND staled_at IS NULL
            )
            INSERT INTO ChatInferenceDatapoint
            SELECT *
            REPLACE (
                now64() AS updated_at,
                now64() AS staled_at
            )
            FROM existing_chat_datapoints;
            "
        );

        let json_query = format!(
            r"
            WITH existing_json_datapoints AS (
                SELECT *
                FROM JsonInferenceDatapoint FINAL
                WHERE dataset_name = {{dataset_name:String}}
                {datapoint_ids_filter_clause}
                AND staled_at IS NULL
            )
            INSERT INTO JsonInferenceDatapoint
            SELECT *
            REPLACE (
                now64() AS updated_at,
                now64() AS staled_at
            )
            FROM existing_json_datapoints;
            "
        );
        let query_params = HashMap::from([("dataset_name", dataset_name)]);

        let (chat_result, json_result) = try_join!(
            self.run_query_synchronous(chat_query, &query_params),
            self.run_query_synchronous(json_query, &query_params)
        )?;
        Ok(chat_result.metadata.written_rows + json_result.metadata.written_rows)
    }

    /// Inserts a batch of datapoints into the database. Internally, separate chat and JSON datapoints and write them to the appropriate tables. Note that this is not very atomic: the Chat table and Json table updates are not rolled back if one fails.
    ///
    /// Returns the number of rows written.
    async fn insert_datapoints(&self, datapoints: &[DatapointInsert]) -> Result<u64, Error> {
        // Separate chat and JSON datapoints
        let mut chat_datapoints: Vec<&ChatInferenceDatapointInsert> = Vec::new();
        let mut json_datapoints: Vec<&JsonInferenceDatapointInsert> = Vec::new();

        for datapoint in datapoints {
            match datapoint {
                DatapointInsert::Chat(chat) => chat_datapoints.push(chat),
                DatapointInsert::Json(json) => json_datapoints.push(json),
            }
        }

        let mut written_rows = 0;

        let (chat_written_rows, json_written_rows) = futures::join!(
            self.insert_chat_datapoints_internal(&chat_datapoints),
            self.insert_json_datapoints_internal(&json_datapoints),
        );

        written_rows += chat_written_rows?;
        written_rows += json_written_rows?;
        Ok(written_rows)
    }
}

/// Converts a vec of OrderBy terms to the correct ClickHouse ORDER BY clauses.
fn get_order_by_clause(
    order_by: Option<&Vec<DatapointOrderBy>>,
    search_query_experimental: Option<&String>,
) -> Result<String, Error> {
    let Some(order_by_vec) = order_by else {
        return Ok("ORDER BY updated_at DESC, id DESC".to_string());
    };
    if order_by_vec.is_empty() {
        return Ok("ORDER BY updated_at DESC, id DESC".to_string());
    }

    // Validate that if SearchRelevance is used, search_query_experimental must be present
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

    // Generate ORDER BY SQL for datapoints
    let order_parts: Vec<String> = order_by_vec
        .iter()
        .map(|order_spec| {
            let column = match order_spec.term {
                DatapointOrderByTerm::Timestamp => "updated_at".to_string(),
                DatapointOrderByTerm::SearchRelevance => "total_term_frequency".to_string(),
            };
            let direction = match order_spec.direction {
                OrderDirection::Asc => "ASC",
                OrderDirection::Desc => "DESC",
            };
            format!("{column} {direction}")
        })
        .collect();

    Ok(format!("ORDER BY {}", order_parts.join(", ")))
}

/// Escapes a string for JSON without quotes.
/// This is used to escape the text query when doing a substring match on input and output strings, because
/// input and output strings are JSON-escaped in ClickHouse.
fn json_escape_string_without_quotes(s: &str) -> Result<String, Error> {
    let mut json_escaped = serde_json::to_string(s)?;
    json_escaped.remove(0);
    json_escaped.pop();
    Ok(json_escaped)
}

impl ClickHouseConnectionInfo {
    /// Internal helper: Puts chat datapoints into the database
    /// Returns the number of rows written
    async fn insert_chat_datapoints_internal(
        &self,
        datapoints: &[&ChatInferenceDatapointInsert],
    ) -> Result<u64, Error> {
        if datapoints.is_empty() {
            return Ok(0);
        }
        for datapoint in datapoints {
            validate_dataset_name(&datapoint.dataset_name)?;
        }

        let serialized_datapoints: Vec<String> =
            datapoints.iter().map(serde_json::to_string).try_collect()?;

        let query = r"
        INSERT INTO ChatInferenceDatapoint
        (
            dataset_name,
            function_name,
            name,
            id,
            episode_id,
            input,
            output,
            tool_params,
            dynamic_provider_tools,
            dynamic_tools,
            allowed_tools,
            parallel_tool_calls,
            tool_choice,
            tags,
            auxiliary,
            is_deleted,
            is_custom,
            source_inference_id,
            updated_at,
            staled_at
        )
        SELECT
            new_data.dataset_name,
            new_data.function_name,
            new_data.name,
            new_data.id,
            new_data.episode_id,
            new_data.input,
            new_data.output,
            new_data.tool_params,
            new_data.dynamic_provider_tools,
            new_data.dynamic_tools,
            new_data.allowed_tools,
            new_data.parallel_tool_calls,
            new_data.tool_choice,
            new_data.tags,
            new_data.auxiliary,
            new_data.is_deleted,
            new_data.is_custom,
            new_data.source_inference_id,
            now64() as updated_at,
            new_data.staled_at
        FROM new_data
        ";

        let external_data = ExternalDataInfo {
            external_data_name: "new_data".to_string(),
            structure: "dataset_name LowCardinality(String), function_name LowCardinality(String), name Nullable(String), id UUID, episode_id Nullable(UUID), input String, output Nullable(String), tool_params String, dynamic_tools Array(String), dynamic_provider_tools Array(String), allowed_tools Nullable(String), tool_choice Nullable(String), parallel_tool_calls Nullable(bool), tags Map(String, String), auxiliary String, is_deleted Bool, is_custom Bool, source_inference_id Nullable(UUID), staled_at Nullable(String)".to_string(),
            format: "JSONEachRow".to_string(),
            data: serialized_datapoints.join("\n"),
        };
        let result = self
            .run_query_with_external_data(external_data, query.to_string())
            .await?;
        Ok(result.metadata.written_rows)
    }

    /// Internal helper: Puts JSON datapoints into the database
    /// Returns the number of rows written
    async fn insert_json_datapoints_internal(
        &self,
        datapoints: &[&JsonInferenceDatapointInsert],
    ) -> Result<u64, Error> {
        if datapoints.is_empty() {
            return Ok(0);
        }
        for datapoint in datapoints {
            validate_dataset_name(&datapoint.dataset_name)?;
        }

        let serialized_datapoints: Vec<String> =
            datapoints.iter().map(serde_json::to_string).try_collect()?;

        let query = r"
        INSERT INTO JsonInferenceDatapoint
        (
            dataset_name,
            function_name,
            id,
            episode_id,
            input,
            output,
            output_schema,
            tags,
            auxiliary,
            is_deleted,
            updated_at,
            staled_at,
            source_inference_id,
            is_custom,
            name
        )
        SELECT
            new_data.dataset_name,
            new_data.function_name,
            new_data.id,
            new_data.episode_id,
            new_data.input,
            new_data.output,
            new_data.output_schema,
            new_data.tags,
            new_data.auxiliary,
            new_data.is_deleted,
            now64() as updated_at,
            new_data.staled_at,
            new_data.source_inference_id,
            new_data.is_custom,
            new_data.name
        FROM new_data
        ";

        let external_data = ExternalDataInfo {
            external_data_name: "new_data".to_string(),
            structure: "dataset_name LowCardinality(String), function_name LowCardinality(String), id UUID, episode_id Nullable(UUID), input String, output Nullable(String), output_schema Nullable(String), tags Map(String, String), auxiliary String, is_deleted Bool, is_custom Bool, source_inference_id Nullable(UUID), staled_at Nullable(String), name Nullable(String)".to_string(),
            format: "JSONEachRow".to_string(),
            data: serialized_datapoints.join("\n"),
        };
        let result = self
            .run_query_with_external_data(external_data, query.to_string())
            .await?;
        Ok(result.metadata.written_rows)
    }
}

/// Constructs a SELECT query for either the Chat or JSON dataset table.
/// This is used to select inferences that will be inserted into a dataset.
fn build_select_inferences_matching_dataset_subquery(
    params: &DatasetQueryParams,
) -> Result<(String, HashMap<String, String>), Error> {
    // Validate: if variant_name is provided, function_name must also be provided.
    if params.variant_name.is_some() && params.function_name.is_none() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "If variant_name is provided, function_name must also be provided."
                .to_string(),
        }));
    }

    // Select the appropriate table based on inference type.
    let source_table_name = match params.inference_type {
        DatapointKind::Chat => "ChatInference",
        DatapointKind::Json => "JsonInference",
    };

    // Determine the output field based on output_source
    let output_field = match params.output_source {
        DatasetOutputSource::Demonstration => "demo.value as output".to_string(),
        DatasetOutputSource::None => "NULL AS output".to_string(),
        DatasetOutputSource::Inference => "output".to_string(),
    };

    // Start building the base query.
    let type_specific_fields = match params.inference_type {
        DatapointKind::Chat => {
            r"tool_params,
dynamic_tools,
dynamic_provider_tools,
parallel_tool_calls,
tool_choice,
allowed_tools,
        "
        }
        DatapointKind::Json => "output_schema,",
    };
    let mut query = format!(
        "SELECT
            NULL as dataset_name,
            function_name,
            NULL as name,
            id,
            episode_id,
            input,
            {output_field},
            {type_specific_fields}
            tags,
            NULL as staled_at,
            id as source_inference_id,
            false as is_custom,
            '' AS auxiliary
        FROM {source_table_name}"
    );

    // Prepare WHERE clause array and query parameters.
    let mut where_clauses: Vec<String> = Vec::new();
    let mut query_params: HashMap<String, String> = HashMap::new();

    // Add condition for function_name if provided.
    if let Some(function_name) = &params.function_name {
        where_clauses.push("function_name = {function_name:String}".to_string());
        query_params.insert("function_name".to_string(), function_name.clone());
    }

    // Add condition for variant_name if provided.
    if let Some(variant_name) = &params.variant_name {
        where_clauses.push("variant_name = {variant_name:String}".to_string());
        query_params.insert("variant_name".to_string(), variant_name.clone());
    }

    // Add extra_params to query_params
    if let Some(extra_params) = &params.extra_params {
        for (k, v) in extra_params {
            if query_params.contains_key(k) {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Extra parameter {k} is already in use"),
                }));
            }
            query_params.insert(k.clone(), v.clone());
        }
    }

    // Join with Metric Filter
    if let Some(metric_filter) = &params.metric_filter {
        let feedback_table = metric_filter.metric_type.to_clickhouse_table_name();
        let operator_str = metric_filter.operator.to_clickhouse_operator();
        let join_on_field = metric_filter.join_on.inference_column_name();

        query += &format!(
            " JOIN (
                SELECT
                    target_id,
                    value,
                    ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
                FROM {feedback_table}
                WHERE metric_name = {{metric_name:String}}
                    AND value {operator_str} {{metric_threshold:Float}}
                ) AS feedback
                ON {source_table_name}.{join_on_field} = feedback.target_id
                    AND feedback.rn = 1"
        );

        query_params.insert("metric_name".to_string(), metric_filter.metric.clone());
        query_params.insert(
            "metric_threshold".to_string(),
            metric_filter.threshold.to_string(),
        );
    }

    // Join with Demonstration Feedback
    if matches!(params.output_source, DatasetOutputSource::Demonstration) {
        query += &format!(
            " JOIN (
                SELECT
                    inference_id,
                    value,
                    ROW_NUMBER() OVER (PARTITION BY inference_id ORDER BY timestamp DESC) as rn
                FROM DemonstrationFeedback
                ) AS demo
                ON {source_table_name}.id = demo.inference_id
                    AND demo.rn = 1"
        );
    }

    // Append any extra WHERE clauses provided by the caller.
    if let Some(extra_where) = &params.extra_where {
        if !extra_where.is_empty() {
            where_clauses.extend(extra_where.iter().cloned());
        }
    }

    // If any WHERE conditions have been added, append them to the query.
    if !where_clauses.is_empty() {
        query += " WHERE ";
        query += &where_clauses.join(" AND ");
    }

    // Append LIMIT and OFFSET clauses if provided.
    if let Some(limit) = params.limit {
        query += " LIMIT {limit:UInt32}";
        query_params.insert("limit".to_string(), limit.to_string());
    }
    if let Some(offset) = params.offset {
        query += " OFFSET {offset:UInt32}";
        query_params.insert("offset".to_string(), offset.to_string());
    }

    Ok((query, query_params))
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Arc;
    use uuid::Uuid;

    use crate::config::{MetricConfigLevel, MetricConfigType};
    use crate::db::clickhouse::clickhouse_client::MockClickHouseClient;
    use crate::db::clickhouse::query_builder::test_util::{
        assert_query_contains, assert_query_does_not_contain,
    };
    use crate::db::clickhouse::query_builder::{
        DatapointFilter, FloatComparisonOperator, TagComparisonOperator, TagFilter,
    };
    use crate::db::clickhouse::ClickHouseResponse;
    use crate::db::clickhouse::ClickHouseResponseMetadata;
    use crate::db::datasets::{
        ChatInferenceDatapointInsert, JsonInferenceDatapointInsert, MetricFilter,
    };
    use crate::endpoints::datasets::v1::types::DatapointOrderBy;
    use crate::inference::types::{ContentBlockChatOutput, JsonInferenceOutput, StoredInput, Text};
    use crate::tool::{
        AllowedTools, AllowedToolsChoice, FunctionTool, Tool, ToolCallConfigDatabaseInsert,
        ToolChoice,
    };

    use super::*;

    fn default_params(datapoint_kind: DatapointKind) -> DatasetQueryParams {
        DatasetQueryParams {
            inference_type: datapoint_kind,
            function_name: None,
            dataset_name: None,
            variant_name: None,
            extra_where: None,
            extra_params: None,
            metric_filter: None,
            output_source: DatasetOutputSource::Inference,
            limit: None,
            offset: None,
        }
    }

    #[test]
    fn test_basic_chat_query() {
        let params = default_params(DatapointKind::Chat);
        let (query, query_params) =
            build_select_inferences_matching_dataset_subquery(&params).unwrap();

        assert_query_contains(
            &query,
            "
            SELECT
                NULL as dataset_name,
                function_name,
                NULL as name,
                id,
                episode_id,
                input,
                output,
                tool_params,
                dynamic_tools,
                dynamic_provider_tools,
                parallel_tool_calls,
                tool_choice,
                allowed_tools,
                tags,
                NULL as staled_at,
                id as source_inference_id,
                false as is_custom,
                '' AS auxiliary
            FROM ChatInference
        ",
        );
        assert!(!query.contains("output_schema")); // Chat doesn't have output_schema
        assert!(!query.contains("WHERE")); // No filters
        assert!(query_params.is_empty());
    }

    #[test]
    fn test_basic_json_query() {
        let params = default_params(DatapointKind::Json);
        let (query, query_params) =
            build_select_inferences_matching_dataset_subquery(&params).unwrap();

        assert_query_contains(
            &query,
            "
            SELECT
                NULL as dataset_name,
                function_name,
                NULL as name,
                id,
                episode_id,
                input,
                output,
                output_schema,
                tags,
                NULL as staled_at,
                id as source_inference_id,
                false as is_custom,
                '' AS auxiliary
            FROM JsonInference
        ",
        );
        assert!(!query.contains("tool_params")); // JSON doesn't have tool_params
        assert!(!query.contains("WHERE")); // No filters
        assert!(query_params.is_empty());
    }

    #[test]
    fn test_function_name_filter() {
        let mut params = default_params(DatapointKind::Chat);
        params.function_name = Some("my_function".to_string());

        let (query, query_params) =
            build_select_inferences_matching_dataset_subquery(&params).unwrap();

        assert_query_contains(
            &query,
            "
            FROM ChatInference
            WHERE function_name = {function_name:String}
        ",
        );
        assert_eq!(
            query_params.get("function_name"),
            Some(&"my_function".to_string())
        );
    }

    #[test]
    fn test_variant_name_filter() {
        let mut params = default_params(DatapointKind::Chat);
        params.function_name = Some("my_function".to_string());
        params.variant_name = Some("my_variant".to_string());

        let (query, query_params) =
            build_select_inferences_matching_dataset_subquery(&params).unwrap();

        assert_query_contains(
            &query,
            "
            FROM ChatInference
            WHERE function_name = {function_name:String}
              AND variant_name = {variant_name:String}
        ",
        );
        assert_eq!(
            query_params.get("function_name"),
            Some(&"my_function".to_string())
        );
        assert_eq!(
            query_params.get("variant_name"),
            Some(&"my_variant".to_string())
        );
    }

    #[test]
    fn test_variant_name_without_function_name_fails() {
        let mut params = default_params(DatapointKind::Chat);
        params.variant_name = Some("my_variant".to_string());

        let result = build_select_inferences_matching_dataset_subquery(&params);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("variant_name"));
        assert!(err.to_string().contains("function_name"));
    }

    #[test]
    fn test_output_source_none() {
        let mut params = default_params(DatapointKind::Chat);
        params.output_source = DatasetOutputSource::None;

        let (query, _) = build_select_inferences_matching_dataset_subquery(&params).unwrap();

        assert_query_contains(&query, "input, NULL AS output, tool_params");
    }

    #[test]
    fn test_output_source_inference() {
        let mut params = default_params(DatapointKind::Chat);
        params.output_source = DatasetOutputSource::Inference;

        let (query, _) = build_select_inferences_matching_dataset_subquery(&params).unwrap();

        // Should just select "output" field directly
        assert_query_contains(&query, "input, output, tool_params");
        assert!(!query.contains("NULL AS output"));
        assert!(!query.contains("demo.value"));
    }

    #[test]
    fn test_output_source_demonstration() {
        let mut params = default_params(DatapointKind::Chat);
        params.output_source = DatasetOutputSource::Demonstration;

        let (query, _) = build_select_inferences_matching_dataset_subquery(&params).unwrap();

        assert_query_contains(&query, "input, demo.value as output, tool_params");
        assert_query_contains(
            &query,
            "
            JOIN (
                SELECT
                    inference_id,
                    value,
                    ROW_NUMBER() OVER (PARTITION BY inference_id ORDER BY timestamp DESC) as rn
                FROM DemonstrationFeedback
            ) AS demo ON ChatInference.id = demo.inference_id AND demo.rn = 1
        ",
        );
    }

    #[test]
    fn test_metric_filter_float_greater_than_id() {
        let mut params = default_params(DatapointKind::Chat);
        params.metric_filter = Some(MetricFilter {
            metric: "accuracy".to_string(),
            metric_type: MetricConfigType::Float,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.8,
            join_on: MetricConfigLevel::Inference,
        });

        let (query, query_params) =
            build_select_inferences_matching_dataset_subquery(&params).unwrap();

        assert_query_contains(
            &query,
            "
            JOIN (
                SELECT
                    target_id,
                    value,
                    ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
                FROM FloatMetricFeedback
                WHERE metric_name = {metric_name:String}
                  AND value > {metric_threshold:Float}
            ) AS feedback ON ChatInference.id = feedback.target_id AND feedback.rn = 1
        ",
        );
        assert_eq!(
            query_params.get("metric_name"),
            Some(&"accuracy".to_string())
        );
        assert_eq!(
            query_params.get("metric_threshold"),
            Some(&"0.8".to_string())
        );
    }

    #[test]
    fn test_metric_filter_boolean_less_than_episode() {
        let mut params = default_params(DatapointKind::Json);
        params.metric_filter = Some(MetricFilter {
            metric: "success".to_string(),
            metric_type: MetricConfigType::Boolean,
            operator: FloatComparisonOperator::LessThan,
            threshold: 1.0,
            join_on: MetricConfigLevel::Episode,
        });

        let (query, query_params) =
            build_select_inferences_matching_dataset_subquery(&params).unwrap();

        assert_query_contains(
            &query,
            "
            JOIN (
                SELECT
                    target_id,
                    value,
                    ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
                FROM BooleanMetricFeedback
                WHERE metric_name = {metric_name:String}
                  AND value < {metric_threshold:Float}
            ) AS feedback ON JsonInference.episode_id = feedback.target_id AND feedback.rn = 1
        ",
        );
        assert_eq!(
            query_params.get("metric_name"),
            Some(&"success".to_string())
        );
        assert_eq!(query_params.get("metric_threshold"), Some(&"1".to_string()));
    }

    #[test]
    fn test_extra_where_clauses() {
        let mut params = default_params(DatapointKind::Chat);
        params.extra_where = Some(vec![
            "created_at > '2024-01-01'".to_string(),
            "episode_id IS NOT NULL".to_string(),
        ]);

        let (query, _) = build_select_inferences_matching_dataset_subquery(&params).unwrap();

        assert_query_contains(
            &query,
            "
            WHERE created_at > '2024-01-01'
              AND episode_id IS NOT NULL
        ",
        );
    }

    #[test]
    fn test_limit_and_offset() {
        let mut params = default_params(DatapointKind::Chat);
        params.limit = Some(50);
        params.offset = Some(100);

        let (query, query_params) =
            build_select_inferences_matching_dataset_subquery(&params).unwrap();

        assert_query_contains(&query, "LIMIT {limit:UInt32}");
        assert_query_contains(&query, "OFFSET {offset:UInt32}");
        assert_eq!(query_params.get("limit"), Some(&"50".to_string()));
        assert_eq!(query_params.get("offset"), Some(&"100".to_string()));
    }

    #[test]
    fn test_complex_query_with_all_features() {
        let mut params = default_params(DatapointKind::Json);
        params.function_name = Some("extract_entities".to_string());
        params.variant_name = Some("v1".to_string());
        params.output_source = DatasetOutputSource::Demonstration;
        params.metric_filter = Some(MetricFilter {
            metric: "f1_score".to_string(),
            metric_type: MetricConfigType::Float,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.9,
            join_on: MetricConfigLevel::Inference,
        });
        params.extra_where = Some(vec!["created_at > '2024-01-01'".to_string()]);
        params.limit = Some(10);
        params.offset = Some(5);

        let (query, query_params) =
            build_select_inferences_matching_dataset_subquery(&params).unwrap();

        // Verify SELECT clause with demonstration output
        assert_query_contains(&query, "demo.value as output");

        // Verify metric filter JOIN
        assert_query_contains(
            &query,
            "
            JOIN (
                SELECT
                    target_id,
                    value,
                    ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
                FROM FloatMetricFeedback
                WHERE metric_name = {metric_name:String}
                  AND value > {metric_threshold:Float}
            ) AS feedback ON JsonInference.id = feedback.target_id AND feedback.rn = 1
        ",
        );

        // Verify demonstration JOIN
        assert_query_contains(
            &query,
            "
            JOIN (
                SELECT
                    inference_id,
                    value,
                    ROW_NUMBER() OVER (PARTITION BY inference_id ORDER BY timestamp DESC) as rn
                FROM DemonstrationFeedback
            ) AS demo ON JsonInference.id = demo.inference_id AND demo.rn = 1
        ",
        );

        // Verify WHERE clause
        assert_query_contains(
            &query,
            "
            WHERE function_name = {function_name:String}
              AND variant_name = {variant_name:String}
              AND created_at > '2024-01-01'
        ",
        );

        // Verify LIMIT and OFFSET
        assert_query_contains(&query, "LIMIT {limit:UInt32}");
        assert_query_contains(&query, "OFFSET {offset:UInt32}");

        // Verify all parameters are set
        assert_eq!(
            query_params.get("function_name"),
            Some(&"extract_entities".to_string())
        );
        assert_eq!(query_params.get("variant_name"), Some(&"v1".to_string()));
        assert_eq!(
            query_params.get("metric_name"),
            Some(&"f1_score".to_string())
        );
        assert_eq!(
            query_params.get("metric_threshold"),
            Some(&"0.9".to_string())
        );
        assert_eq!(query_params.get("limit"), Some(&"10".to_string()));
        assert_eq!(query_params.get("offset"), Some(&"5".to_string()));
    }

    #[test]
    fn test_select_fields_for_chat() {
        let params = default_params(DatapointKind::Chat);
        let (query, _) = build_select_inferences_matching_dataset_subquery(&params).unwrap();

        // Verify all expected fields are present
        assert!(query.contains("NULL as dataset_name"));
        assert!(query.contains("function_name"));
        assert!(query.contains("NULL as name"));
        assert!(query.contains("id"));
        assert!(query.contains("episode_id"));
        assert!(query.contains("input"));
        assert!(query.contains("output"));
        assert!(query.contains("tool_params"));
        assert!(query.contains("tags"));
        assert!(query.contains("NULL as staled_at"));
        assert!(query.contains("id as source_inference_id"));
        assert!(query.contains("false as is_custom"));
        assert!(query.contains("'' AS auxiliary"));
    }

    #[test]
    fn test_select_fields_for_json() {
        let params = default_params(DatapointKind::Json);
        let (query, _) = build_select_inferences_matching_dataset_subquery(&params).unwrap();

        // Verify all expected fields are present
        assert!(query.contains("NULL as dataset_name"));
        assert!(query.contains("function_name"));
        assert!(query.contains("NULL as name"));
        assert!(query.contains("id"));
        assert!(query.contains("episode_id"));
        assert!(query.contains("input"));
        assert!(query.contains("output"));
        assert!(query.contains("output_schema"));
        assert!(query.contains("tags"));
        assert!(query.contains("NULL as staled_at"));
        assert!(query.contains("id as source_inference_id"));
        assert!(query.contains("false as is_custom"));
        assert!(query.contains("'' AS auxiliary"));
    }

    #[test]
    fn test_extra_params_string() {
        let mut params = default_params(DatapointKind::Chat);
        params.extra_where = Some(vec!["user_id = {user_id:String}".to_string()]);
        let mut extra_params = HashMap::new();
        extra_params.insert("user_id".to_string(), "user123".to_string());
        params.extra_params = Some(extra_params);

        let (query, query_params) =
            build_select_inferences_matching_dataset_subquery(&params).unwrap();

        assert_query_contains(&query, "WHERE user_id = {user_id:String}");
        assert_eq!(query_params.get("user_id"), Some(&"user123".to_string()));
    }

    #[test]
    fn test_extra_params_number() {
        let mut params = default_params(DatapointKind::Json);
        params.extra_where = Some(vec!["score > {min_score:Float}".to_string()]);
        let mut extra_params = HashMap::new();
        extra_params.insert("min_score".to_string(), "0.75".to_string());
        params.extra_params = Some(extra_params);

        let (query, query_params) =
            build_select_inferences_matching_dataset_subquery(&params).unwrap();

        assert_query_contains(&query, "WHERE score > {min_score:Float}");
        assert_eq!(query_params.get("min_score"), Some(&"0.75".to_string()));
    }

    #[test]
    fn test_extra_params_multiple() {
        let mut params = default_params(DatapointKind::Chat);
        params.extra_where = Some(vec![
            "user_id = {user_id:String}".to_string(),
            "score > {min_score:Float}".to_string(),
        ]);
        let mut extra_params = HashMap::new();
        extra_params.insert("user_id".to_string(), "user123".to_string());
        extra_params.insert("min_score".to_string(), "0.5".to_string());
        params.extra_params = Some(extra_params);

        let (query, query_params) =
            build_select_inferences_matching_dataset_subquery(&params).unwrap();

        assert_query_contains(
            &query,
            "WHERE user_id = {user_id:String} AND score > {min_score:Float}",
        );
        assert_eq!(query_params.get("user_id"), Some(&"user123".to_string()));
        assert_eq!(query_params.get("min_score"), Some(&"0.5".to_string()));
    }

    #[test]
    fn test_extra_params_reserved_name_conflict() {
        let mut params = default_params(DatapointKind::Chat);
        params.function_name = Some("my_function".to_string());
        let mut extra_params = HashMap::new();
        // Try to override the function_name parameter
        extra_params.insert("function_name".to_string(), "other_function".to_string());
        params.extra_params = Some(extra_params);

        let result = build_select_inferences_matching_dataset_subquery(&params);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("function_name"));
        assert!(err.to_string().contains("is already in use"));
    }

    #[tokio::test]
    async fn test_count_rows_for_dataset_executes() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "SELECT toUInt32(count()) as count FROM (SELECT");
                // Not asserting on the subquery
                assert_query_contains(query, "WHERE function_name = {function_name:String})");
                assert_query_does_not_contain(query, "FORMAT JSONEachRow");

                assert_eq!(parameters.get("function_name"), Some(&"test_function"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from("2"),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let mut params = default_params(DatapointKind::Chat);
        params.function_name = Some("test_function".to_string());

        let result = conn.count_rows_for_dataset(&params).await.unwrap();

        assert_eq!(result, 2, "Should return 2 rows");
    }

    #[tokio::test]
    async fn test_count_rows_rejects_limit() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .times(0);
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let mut params = default_params(DatapointKind::Chat);
        params.limit = Some(10);

        let result = conn.count_rows_for_dataset(&params).await;

        assert!(result.is_err(), "Should reject params with limit");
    }

    #[tokio::test]
    async fn test_count_rows_rejects_offset() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .times(0);
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let mut params = default_params(DatapointKind::Chat);
        params.offset = Some(10);

        let result = conn.count_rows_for_dataset(&params).await;

        assert!(result.is_err(), "Should reject params with offset");
    }

    #[tokio::test]
    async fn test_insert_rows_for_dataset_with_chat_inferences_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(
                    query,
                    "
                INSERT INTO {datapoint_table:Identifier}
                    SELECT {dataset_name:String} as dataset_name,
                    subquery.function_name as function_name,
                    generateUUIDv7() as id,
                    subquery.episode_id as episode_id,
                    subquery.input as input,
                    subquery.output as output,
                    subquery.tool_params,
                    subquery.dynamic_tools,
                    subquery.dynamic_provider_tools,
                    subquery.parallel_tool_calls,
                    subquery.tool_choice,
                    subquery.allowed_tools,
                    subquery.tags as tags,
                    subquery.auxiliary as auxiliary,
                    false as is_deleted,
                    now64() as updated_at,
                    null as staled_at,
                    subquery.id as source_inference_id,
                    false as is_custom,
                    subquery.name as name
                FROM (",
                );
                assert_query_contains(
                    query,
                    "
                ) AS subquery
                LEFT JOIN {datapoint_table:Identifier} as existing FINAL
                ON {dataset_name:String} = existing.dataset_name
                   AND subquery.function_name = existing.function_name
                   AND subquery.id = existing.source_inference_id
                   AND existing.staled_at IS NULL
                WHERE existing.source_inference_id IS NULL",
                );
                assert_query_does_not_contain(query, "subquery.output_schema");
                assert_query_does_not_contain(query, "FORMAT JSONEachRow");

                assert_eq!(parameters.get("dataset_name"), Some(&"my_dataset"));
                assert_eq!(
                    parameters.get("datapoint_table"),
                    Some(&"ChatInferenceDatapoint")
                );

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 10,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let mut params = default_params(DatapointKind::Chat);
        params.dataset_name = Some("my_dataset".to_string());

        let rows_inserted = conn.insert_rows_for_dataset(&params).await.unwrap();

        assert_eq!(rows_inserted, 10);
    }

    #[tokio::test]
    async fn test_insert_rows_for_dataset_with_json_inferences_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "INSERT INTO {datapoint_table:Identifier}");
                assert_query_contains(query, "subquery.output_schema");
                assert_query_does_not_contain(query, "subquery.tool_params");
                assert_eq!(parameters.get("dataset_name"), Some(&"my_dataset"));
                assert_eq!(
                    parameters.get("datapoint_table"),
                    Some(&"JsonInferenceDatapoint")
                );

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 20,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let mut params = default_params(DatapointKind::Json);
        params.dataset_name = Some("my_dataset".to_string());

        let rows_inserted = conn.insert_rows_for_dataset(&params).await.unwrap();

        assert_eq!(rows_inserted, 20);
    }

    #[tokio::test]
    async fn test_insert_rows_requires_dataset_name() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .times(0);
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let params = default_params(DatapointKind::Chat);
        // No dataset_name provided

        let result = conn.insert_rows_for_dataset(&params).await;

        assert!(
            result.is_err(),
            "Should reject params if dataset_name is not provided"
        );
    }

    #[tokio::test]
    async fn test_insert_rows_rejects_reserved_dataset_names_builder() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .times(0);
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let mut params = default_params(DatapointKind::Chat);
        params.dataset_name = Some("builder".to_string());

        let result = conn.insert_rows_for_dataset(&params).await;

        assert!(result.is_err(), "Should reject dataset_name 'builder'");
    }

    #[tokio::test]
    async fn test_insert_rows_rejects_reserved_dataset_names_tensorzero_prefix() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .times(0);
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let mut params = default_params(DatapointKind::Chat);
        params.dataset_name = Some("tensorzero::openai".to_string());

        let result = conn.insert_rows_for_dataset(&params).await;

        assert!(
            result.is_err(),
            "Should reject dataset_name starting with tensorzero::"
        );
    }

    #[tokio::test]
    async fn test_get_dataset_metadata_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "
                    SELECT
                    dataset_name,
                    toUInt32(sum(count)) AS count,
                    formatDateTime(max(last_updated), '%Y-%m-%dT%H:%i:%SZ') AS last_updated
                    FROM (
                        SELECT
                            dataset_name,
                            toUInt32(count()) AS count,
                            max(updated_at) AS last_updated
                        FROM ChatInferenceDatapoint
                        FINAL
                        WHERE staled_at IS NULL
                        AND function_name = {function_name:String}
                        GROUP BY dataset_name
                        UNION ALL
                        SELECT
                            dataset_name,
                            toUInt32(count()) AS count,
                            max(updated_at) AS last_updated
                        FROM JsonInferenceDatapoint
                        FINAL
                        WHERE staled_at IS NULL
                        AND function_name = {function_name:String}
                        GROUP BY dataset_name
                )
                GROUP BY dataset_name
                ORDER BY last_updated DESC
                LIMIT {limit:UInt32}
                OFFSET {offset:UInt32}
                FORMAT JSONEachRow");

                assert_eq!(parameters.get("function_name"), Some(&"test_function"));
                assert_eq!(parameters.get("limit"), Some(&"10"));
                assert_eq!(parameters.get("offset"), Some(&"20"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"
                    {"dataset_name": "test-dataset-1","count": 10,"last_updated": "2021-01-01T00:00:00Z"}
                    {"dataset_name": "test-dataset-2","count": 20,"last_updated": "2021-01-01T00:00:00Z"}
                    "#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let params = GetDatasetMetadataParams {
            function_name: Some("test_function".to_string()),
            limit: Some(10),
            offset: Some(20),
        };

        let rows = conn.get_dataset_metadata(&params).await.unwrap();

        assert_eq!(rows.len(), 2, "Should return 2 datasets");
        assert_eq!(rows[0].dataset_name, "test-dataset-1");
        assert_eq!(rows[1].dataset_name, "test-dataset-2");
    }

    #[tokio::test]
    async fn test_get_dataset_metadata_without_filters() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_does_not_contain(query, "AND function_name = {function_name:String}");
                assert_query_does_not_contain(query, "LIMIT {limit:UInt32}");
                assert_query_does_not_contain(query, "OFFSET {offset:UInt32}");

                assert!(
                    !parameters.contains_key("function_name"),
                    "Should not provide function_name as a query parameter"
                );
                assert!(
                    !parameters.contains_key("limit"),
                    "Should not provide limit as a query parameter"
                );
                assert!(
                    !parameters.contains_key("offset"),
                    "Should not provide offset as a query parameter"
                );

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

        let params = GetDatasetMetadataParams {
            function_name: None,
            limit: None,
            offset: None,
        };

        let metadata = conn.get_dataset_metadata(&params).await.unwrap();

        assert_eq!(metadata.len(), 0);
    }

    #[tokio::test]
    async fn test_count_datasets_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(
                    query,
                    "SELECT toUInt32(uniqExact(dataset_name)) as count FROM (",
                );
                assert_query_contains(
                    query,
                    "SELECT dataset_name FROM ChatInferenceDatapoint FINAL WHERE staled_at IS NULL",
                );
                assert_query_contains(query, "UNION ALL");
                assert_query_contains(
                    query,
                    "SELECT dataset_name FROM JsonInferenceDatapoint FINAL WHERE staled_at IS NULL",
                );
                assert_query_does_not_contain(query, "FORMAT JSONEachRow");
                assert!(
                    parameters.is_empty(),
                    "Should not have any query parameters"
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: "5".to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn.count_datasets().await.unwrap();

        assert_eq!(result, 5, "Should return 5 datasets");
    }

    #[tokio::test]
    async fn test_insert_chat_datapoint_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_with_external_data()
            .withf(|external_data, query| {
                // Verify the query is correct
                assert!(query.contains("INSERT INTO ChatInferenceDatapoint"));

                // Verify the external data structure
                assert_eq!(external_data.external_data_name, "new_data");
                assert_eq!(external_data.format, "JSONEachRow");
                assert!(external_data
                    .structure
                    .contains("dataset_name LowCardinality(String)"));

                // Parse and verify the data
                let actual_row_as_json: serde_json::Value =
                    serde_json::from_str(&external_data.data).unwrap();
                // When tool_params is None, the new Migration 0041 fields are not serialized
                // (due to #[serde(flatten)]), so they won't be in the JSON
                let expected_row_as_json = json!({
                    "dataset_name": "test_dataset",
                    "function_name": "test_function",
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "name": "test_name",
                    "episode_id": "123e4567-e89b-12d3-a456-426614174000",
                    "input": {
                        "messages": [],
                    },
                    "output": [{"type": "text", "text": "response"}],
                    "tags": {"test_tag": "test_value"},
                    "auxiliary": "",
                    "source_inference_id": null,
                    "is_custom": true,
                    "staled_at": null,
                });
                assert_eq!(
                    actual_row_as_json, expected_row_as_json,
                    "Actual ChatInferenceDatapoint should match expected json value"
                );

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        written_rows: 1,
                        read_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let datapoint = ChatInferenceDatapointInsert {
            dataset_name: "test_dataset".to_string(),
            function_name: "test_function".to_string(),
            id: Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap(),
            name: Some("test_name".to_string()),
            episode_id: Some(Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()),
            input: StoredInput {
                system: None,
                messages: vec![],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "response".to_string(),
            })]),
            tool_params: None,
            tags: Some(HashMap::from([(
                "test_tag".to_string(),
                "test_value".to_string(),
            )])),
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };
        assert!(
            conn.insert_datapoints(&[DatapointInsert::Chat(datapoint)])
                .await
                .is_ok(),
            "Should insert chat datapoint successfully"
        );
    }

    #[tokio::test]
    async fn test_insert_chat_datapoint_with_tool_params_executes_successfully() {
        use crate::tool::{
            AllowedTools, AllowedToolsChoice, FunctionTool, Tool, ToolCallConfigDatabaseInsert,
            ToolChoice,
        };

        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_with_external_data()
            .withf(|external_data, query| {
                // Verify the query is correct
                assert!(query.contains("INSERT INTO ChatInferenceDatapoint"));

                // Parse and verify the data includes all Migration 0041 columns
                let actual_row_as_json: serde_json::Value =
                    serde_json::from_str(&external_data.data).unwrap();

                // Verify the new Migration 0041 columns are present with correct values
                assert_eq!(
                    actual_row_as_json["dynamic_tools"],
                    json!([{"type": "function", "description": "Get temperature", "parameters": {"type": "object"}, "name": "get_temperature", "strict": true}])
                );
                assert_eq!(actual_row_as_json["dynamic_provider_tools"], json!([]));
                assert_eq!(
                    actual_row_as_json["allowed_tools"],
                    json!({"tools": ["weather_tool"], "choice": "dynamic_allowed_tools"})
                );
                assert_eq!(actual_row_as_json["tool_choice"], json!("required"));
                assert_eq!(actual_row_as_json["parallel_tool_calls"], json!(true));

                // Verify legacy tool_params field is also present
                assert!(actual_row_as_json.get("tool_params").is_some());

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        written_rows: 1,
                        read_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let datapoint = ChatInferenceDatapointInsert {
            dataset_name: "test_dataset".to_string(),
            function_name: "test_function".to_string(),
            id: Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap(),
            name: Some("test_with_tools".to_string()),
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "response".to_string(),
            })]),
            tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
                vec![Tool::Function(FunctionTool {
                    name: "get_temperature".to_string(),
                    description: "Get temperature".to_string(),
                    parameters: json!({"type": "object"}),
                    strict: true,
                })],
                vec![],
                AllowedTools {
                    tools: ["weather_tool".to_string()].into_iter().collect(),
                    #[expect(deprecated)]
                    choice: AllowedToolsChoice::DynamicAllowedTools,
                },
                ToolChoice::Required,
                Some(true),
            )),
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };
        assert!(
            conn.insert_datapoints(&[DatapointInsert::Chat(datapoint)])
                .await
                .is_ok(),
            "Should insert chat datapoint with tool_params successfully"
        );
    }

    #[tokio::test]
    async fn test_insert_chat_datapoint_with_explicit_allowed_tools_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_with_external_data()
            .withf(|external_data, query| {
                // Verify the query is correct
                assert!(query.contains("INSERT INTO ChatInferenceDatapoint"));

                // Parse and verify the data includes all Migration 0041 columns
                let actual_row_as_json: serde_json::Value =
                    serde_json::from_str(&external_data.data).unwrap();

                // Verify the new Migration 0041 columns are present with correct values
                assert_eq!(
                    actual_row_as_json["dynamic_tools"],
                    json!([{"type": "function", "description": "Get temperature", "parameters": {"type": "object"}, "name": "get_temperature", "strict": true}])
                );
                assert_eq!(actual_row_as_json["dynamic_provider_tools"], json!([]));
                assert_eq!(
                    actual_row_as_json["allowed_tools"],
                    json!({"tools": ["weather_tool"], "choice": "explicit"})
                );
                assert_eq!(actual_row_as_json["tool_choice"], json!("required"));
                assert_eq!(actual_row_as_json["parallel_tool_calls"], json!(true));

                // Verify legacy tool_params field is also present
                assert!(actual_row_as_json.get("tool_params").is_some());

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        written_rows: 1,
                        read_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let datapoint = ChatInferenceDatapointInsert {
            dataset_name: "test_dataset".to_string(),
            function_name: "test_function".to_string(),
            id: Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap(),
            name: Some("test_with_tools".to_string()),
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "response".to_string(),
            })]),
            tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
                vec![Tool::Function(FunctionTool {
                    name: "get_temperature".to_string(),
                    description: "Get temperature".to_string(),
                    parameters: json!({"type": "object"}),
                    strict: true,
                })],
                vec![],
                AllowedTools {
                    tools: ["weather_tool".to_string()].into_iter().collect(),
                    choice: AllowedToolsChoice::Explicit,
                },
                ToolChoice::Required,
                Some(true),
            )),
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };
        assert!(
            conn.insert_datapoints(&[DatapointInsert::Chat(datapoint)])
                .await
                .is_ok(),
            "Should insert chat datapoint with Explicit allowed tool successfully"
        );
    }

    #[tokio::test]
    async fn test_insert_json_datapoint_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_with_external_data()
            .withf(|external_data, query| {
                // Verify the query is correct
                assert!(query.contains("INSERT INTO JsonInferenceDatapoint"));

                // Verify the external data structure
                assert_eq!(external_data.external_data_name, "new_data");
                assert_eq!(external_data.format, "JSONEachRow");
                assert!(external_data
                    .structure
                    .contains("dataset_name LowCardinality(String)"));

                // Parse and verify the data
                let actual_row_as_json: serde_json::Value =
                    serde_json::from_str(&external_data.data).unwrap();
                let expected_row_as_json = json!({
                    "dataset_name": "test_dataset",
                    "function_name": "test_function",
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "name": "test_name",
                    "episode_id": "123e4567-e89b-12d3-a456-426614174000",
                    "input": {
                        "messages": [],
                    },
                    "output": {"raw": "{\"data\":\"extracted\"}", "parsed": {"data":"extracted"}},
                    "output_schema": {"type": "object"},
                    "tags": {"test_tag": "test_value"},
                    "auxiliary": "",
                    "source_inference_id": null,
                    "is_custom": true,
                    "staled_at": null,
                });

                assert_eq!(
                    actual_row_as_json, expected_row_as_json,
                    "Actual JsonInferenceDatapoint should match expected json value"
                );

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        written_rows: 1,
                        read_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let datapoint = JsonInferenceDatapointInsert {
            dataset_name: "test_dataset".to_string(),
            function_name: "test_function".to_string(),
            id: Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap(),
            name: Some("test_name".to_string()),
            episode_id: Some(Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()),
            input: StoredInput {
                system: None,
                messages: vec![],
            },
            output: Some(JsonInferenceOutput {
                parsed: Some(json!({"data":"extracted"})),
                raw: Some("{\"data\":\"extracted\"}".to_string()),
            }),
            output_schema: json!({"type": "object"}),
            tags: Some(HashMap::from([(
                "test_tag".to_string(),
                "test_value".to_string(),
            )])),
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };
        assert!(
            conn.insert_datapoints(&[DatapointInsert::Json(datapoint)])
                .await
                .is_ok(),
            "Should insert json datapoint successfully"
        );
    }

    #[tokio::test]
    async fn test_insert_datapoints_with_only_chat_datapoints() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        // Expect exactly one call to run_query_with_external_data for chat datapoints
        mock_clickhouse_client
            .expect_run_query_with_external_data()
            .times(1)
            .withf(|external_data, query| {
                // Verify the query targets ChatInferenceDatapoint
                assert_query_contains(query,
                    "INSERT INTO ChatInferenceDatapoint (
                    dataset_name,
                    function_name,
                    name,
                    id,
                    episode_id,
                    input,
                    output,
                    tool_params,
                    dynamic_provider_tools,
                    dynamic_tools,
                    allowed_tools,
                    parallel_tool_calls,
                    tool_choice,
                    tags,
                    auxiliary,
                    is_deleted,
                    is_custom,
                    source_inference_id,
                    updated_at,
                    staled_at
                )"
                );
                assert_query_contains(query,
                    "SELECT
                    new_data.dataset_name,
                    new_data.function_name,
                    new_data.name,
                    new_data.id,
                    new_data.episode_id,
                    new_data.input,
                    new_data.output,
                    new_data.tool_params,
                    new_data.dynamic_provider_tools,
                    new_data.dynamic_tools,
                    new_data.allowed_tools,
                    new_data.parallel_tool_calls,
                    new_data.tool_choice,
                    new_data.tags,
                    new_data.auxiliary,
                    new_data.is_deleted,
                    new_data.is_custom,
                    new_data.source_inference_id,
                    now64() as updated_at,
                    new_data.staled_at
                FROM new_data"
                );

                // Verify the external data structure
                assert_eq!(external_data.external_data_name, "new_data");
                assert_eq!(external_data.format, "JSONEachRow");
                assert!(external_data
                    .structure
                    .contains("dataset_name LowCardinality(String), function_name LowCardinality(String), name Nullable(String), id UUID, episode_id Nullable(UUID), input String, output Nullable(String), tool_params String, dynamic_tools Array(String), dynamic_provider_tools Array(String), allowed_tools Nullable(String), tool_choice Nullable(String), parallel_tool_calls Nullable(bool), tags Map(String, String), auxiliary String, is_deleted Bool, is_custom Bool, source_inference_id Nullable(UUID), staled_at Nullable(String)"));
                assert!(!external_data.structure.contains("updated_at"));

                // Parse the data - should contain 3 datapoints separated by newlines
                let lines: Vec<&str> = external_data.data.lines().collect();
                assert_eq!(lines.len(), 3, "Should have 3 chat datapoints");

                // Verify first datapoint
                let first_datapoint: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
                assert_eq!(first_datapoint["dataset_name"], "test_dataset");
                assert_eq!(first_datapoint["function_name"], "test_function_1");
                assert_eq!(
                    first_datapoint["id"],
                    "11111111-1111-1111-1111-111111111111"
                );

                // Verify second datapoint
                let second_datapoint: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
                assert_eq!(second_datapoint["dataset_name"], "test_dataset");
                assert_eq!(second_datapoint["function_name"], "test_function_2");
                assert_eq!(
                    second_datapoint["id"],
                    "22222222-2222-2222-2222-222222222222"
                );

                // Verify third datapoint
                let third_datapoint: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
                assert_eq!(third_datapoint["dataset_name"], "test_dataset");
                assert_eq!(third_datapoint["function_name"], "test_function_3");
                assert_eq!(
                    third_datapoint["id"],
                    "33333333-3333-3333-3333-333333333333"
                );

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        written_rows: 3,
                        read_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let datapoints = vec![
            DatapointInsert::Chat(ChatInferenceDatapointInsert {
                dataset_name: "test_dataset".to_string(),
                function_name: "test_function_1".to_string(),
                id: Uuid::parse_str("11111111-1111-1111-1111-111111111111").unwrap(),
                name: Some("datapoint_1".to_string()),
                episode_id: None,
                input: StoredInput {
                    system: None,
                    messages: vec![],
                },
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "response_1".to_string(),
                })]),
                tool_params: None,
                tags: None,
                auxiliary: String::new(),
                staled_at: None,
                source_inference_id: None,
                is_custom: true,
            }),
            DatapointInsert::Chat(ChatInferenceDatapointInsert {
                dataset_name: "test_dataset".to_string(),
                function_name: "test_function_2".to_string(),
                id: Uuid::parse_str("22222222-2222-2222-2222-222222222222").unwrap(),
                name: Some("datapoint_2".to_string()),
                episode_id: None,
                input: StoredInput {
                    system: None,
                    messages: vec![],
                },
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "response_2".to_string(),
                })]),
                tool_params: None,
                tags: None,
                auxiliary: String::new(),
                staled_at: None,
                source_inference_id: None,
                is_custom: true,
            }),
            DatapointInsert::Chat(ChatInferenceDatapointInsert {
                dataset_name: "test_dataset".to_string(),
                function_name: "test_function_3".to_string(),
                id: Uuid::parse_str("33333333-3333-3333-3333-333333333333").unwrap(),
                name: Some("datapoint_3".to_string()),
                episode_id: None,
                input: StoredInput {
                    system: None,
                    messages: vec![],
                },
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "response_3".to_string(),
                })]),
                tool_params: None,
                tags: None,
                auxiliary: String::new(),
                staled_at: None,
                source_inference_id: None,
                is_custom: true,
            }),
        ];

        let result = conn.insert_datapoints(&datapoints).await;
        assert!(result.is_ok(), "Should insert datapoints successfully");
        assert_eq!(result.unwrap(), 3, "Should return 3 written rows");
    }

    #[tokio::test]
    async fn test_insert_datapoints_with_only_json_datapoints() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        // Expect exactly one call to run_query_with_external_data for JSON datapoints
        mock_clickhouse_client
            .expect_run_query_with_external_data()
            .times(1)
            .withf(|external_data, query| {
                // Verify the query targets JsonInferenceDatapoint
                assert_query_contains(
                    query,
                    "INSERT INTO JsonInferenceDatapoint
                    (
                        dataset_name,
                        function_name,
                        id,
                        episode_id,
                        input,
                        output,
                        output_schema,
                        tags,
                        auxiliary,
                        is_deleted,
                        updated_at,
                        staled_at,
                        source_inference_id,
                        is_custom,
                        name
                    )",
                );
                assert_query_contains(
                    query,
                    "SELECT
                        new_data.dataset_name,
                        new_data.function_name,
                        new_data.id,
                        new_data.episode_id,
                        new_data.input,
                        new_data.output,
                        new_data.output_schema,
                        new_data.tags,
                        new_data.auxiliary,
                        new_data.is_deleted,
                        now64() as updated_at,
                        new_data.staled_at,
                        new_data.source_inference_id,
                        new_data.is_custom,
                        new_data.name
                    FROM new_data",
                );

                // Verify the external data structure
                assert_eq!(external_data.external_data_name, "new_data");
                assert_eq!(external_data.format, "JSONEachRow");
                assert!(external_data
                    .structure
                    .contains("dataset_name LowCardinality(String), function_name LowCardinality(String), id UUID, episode_id Nullable(UUID), input String, output Nullable(String), output_schema Nullable(String), tags Map(String, String), auxiliary String, is_deleted Bool, is_custom Bool, source_inference_id Nullable(UUID), staled_at Nullable(String), name Nullable(String)"));

                // Parse the data - should contain 2 datapoints separated by newlines
                let lines: Vec<&str> = external_data.data.lines().collect();
                assert_eq!(lines.len(), 2, "Should have 2 JSON datapoints");

                // Verify first datapoint
                let first_datapoint: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
                assert_eq!(first_datapoint["dataset_name"], "test_dataset");
                assert_eq!(first_datapoint["function_name"], "json_function_1");
                assert_eq!(
                    first_datapoint["id"],
                    "44444444-4444-4444-4444-444444444444"
                );
                assert!(first_datapoint["output_schema"].is_object());

                // Verify second datapoint
                let second_datapoint: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
                assert_eq!(second_datapoint["dataset_name"], "test_dataset");
                assert_eq!(second_datapoint["function_name"], "json_function_2");
                assert_eq!(
                    second_datapoint["id"],
                    "55555555-5555-5555-5555-555555555555"
                );
                assert!(second_datapoint["output_schema"].is_object());

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        written_rows: 2,
                        read_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let datapoints = vec![
            DatapointInsert::Json(JsonInferenceDatapointInsert {
                dataset_name: "test_dataset".to_string(),
                function_name: "json_function_1".to_string(),
                id: Uuid::parse_str("44444444-4444-4444-4444-444444444444").unwrap(),
                name: Some("json_datapoint_1".to_string()),
                episode_id: None,
                input: StoredInput {
                    system: None,
                    messages: vec![],
                },
                output: Some(JsonInferenceOutput {
                    parsed: Some(json!({"result": "data_1"})),
                    raw: Some("{\"result\":\"data_1\"}".to_string()),
                }),
                output_schema: json!({"type": "object"}),
                tags: None,
                auxiliary: String::new(),
                staled_at: None,
                source_inference_id: None,
                is_custom: true,
            }),
            DatapointInsert::Json(JsonInferenceDatapointInsert {
                dataset_name: "test_dataset".to_string(),
                function_name: "json_function_2".to_string(),
                id: Uuid::parse_str("55555555-5555-5555-5555-555555555555").unwrap(),
                name: Some("json_datapoint_2".to_string()),
                episode_id: None,
                input: StoredInput {
                    system: None,
                    messages: vec![],
                },
                output: Some(JsonInferenceOutput {
                    parsed: Some(json!({"result": "data_2"})),
                    raw: Some("{\"result\":\"data_2\"}".to_string()),
                }),
                output_schema: json!({"type": "object"}),
                tags: None,
                auxiliary: String::new(),
                staled_at: None,
                source_inference_id: None,
                is_custom: true,
            }),
        ];

        let result = conn.insert_datapoints(&datapoints).await;
        assert!(result.is_ok(), "Should insert datapoints successfully");
        assert_eq!(result.unwrap(), 2, "Should return 2 written rows");
    }

    #[tokio::test]
    async fn test_insert_datapoints_with_mixed_datapoints() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        // Expect two calls: one for chat datapoints, one for JSON datapoints
        // The order of calls is determined by futures::join!, which runs concurrently
        // So we need to handle both orders
        mock_clickhouse_client
            .expect_run_query_with_external_data()
            .times(2)
            .withf(|external_data, query| {
                if query.contains("INSERT INTO ChatInferenceDatapoint") {
                    // Verify chat datapoints
                    assert_eq!(external_data.external_data_name, "new_data");
                    assert_eq!(external_data.format, "JSONEachRow");
                    assert!(external_data.structure.contains("tool_params String"));

                    let lines: Vec<&str> = external_data.data.lines().collect();
                    assert_eq!(lines.len(), 2, "Should have 2 chat datapoints");

                    let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
                    assert_eq!(first["dataset_name"], "test_dataset");
                    assert_eq!(first["function_name"], "chat_function_1");

                    let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
                    assert_eq!(second["dataset_name"], "test_dataset");
                    assert_eq!(second["function_name"], "chat_function_2");

                    true
                } else if query.contains("INSERT INTO JsonInferenceDatapoint") {
                    // Verify JSON datapoints
                    assert_eq!(external_data.external_data_name, "new_data");
                    assert_eq!(external_data.format, "JSONEachRow");
                    assert!(external_data
                        .structure
                        .contains("output_schema Nullable(String)"));

                    let lines: Vec<&str> = external_data.data.lines().collect();
                    assert_eq!(lines.len(), 2, "Should have 2 JSON datapoints");

                    let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
                    assert_eq!(first["dataset_name"], "test_dataset");
                    assert_eq!(first["function_name"], "json_function_1");

                    let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
                    assert_eq!(second["dataset_name"], "test_dataset");
                    assert_eq!(second["function_name"], "json_function_2");

                    true
                } else {
                    panic!("Unexpected query: {query}");
                }
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        // Both Chat and Json inserts have 2 rows.
                        written_rows: 2,
                        read_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let datapoints = vec![
            DatapointInsert::Chat(ChatInferenceDatapointInsert {
                dataset_name: "test_dataset".to_string(),
                function_name: "chat_function_1".to_string(),
                id: Uuid::parse_str("66666666-6666-6666-6666-666666666666").unwrap(),
                name: None,
                episode_id: None,
                input: StoredInput {
                    system: None,
                    messages: vec![],
                },
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "chat response 1".to_string(),
                })]),
                tool_params: None,
                tags: None,
                auxiliary: String::new(),
                staled_at: None,
                source_inference_id: None,
                is_custom: false,
            }),
            DatapointInsert::Json(JsonInferenceDatapointInsert {
                dataset_name: "test_dataset".to_string(),
                function_name: "json_function_1".to_string(),
                id: Uuid::parse_str("77777777-7777-7777-7777-777777777777").unwrap(),
                name: None,
                episode_id: None,
                input: StoredInput {
                    system: None,
                    messages: vec![],
                },
                output: Some(JsonInferenceOutput {
                    parsed: Some(json!({"value": 1})),
                    raw: Some("{\"value\":1}".to_string()),
                }),
                output_schema: json!({"type": "object"}),
                tags: None,
                auxiliary: String::new(),
                staled_at: None,
                source_inference_id: None,
                is_custom: false,
            }),
            DatapointInsert::Chat(ChatInferenceDatapointInsert {
                dataset_name: "test_dataset".to_string(),
                function_name: "chat_function_2".to_string(),
                id: Uuid::parse_str("88888888-8888-8888-8888-888888888888").unwrap(),
                name: None,
                episode_id: None,
                input: StoredInput {
                    system: None,
                    messages: vec![],
                },
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "chat response 2".to_string(),
                })]),
                tool_params: None,
                tags: None,
                auxiliary: String::new(),
                staled_at: None,
                source_inference_id: None,
                is_custom: false,
            }),
            DatapointInsert::Json(JsonInferenceDatapointInsert {
                dataset_name: "test_dataset".to_string(),
                function_name: "json_function_2".to_string(),
                id: Uuid::parse_str("99999999-9999-9999-9999-999999999999").unwrap(),
                name: None,
                episode_id: None,
                input: StoredInput {
                    system: None,
                    messages: vec![],
                },
                output: Some(JsonInferenceOutput {
                    parsed: Some(json!({"value": 2})),
                    raw: Some("{\"value\":2}".to_string()),
                }),
                output_schema: json!({"type": "object"}),
                tags: None,
                auxiliary: String::new(),
                staled_at: None,
                source_inference_id: None,
                is_custom: false,
            }),
        ];

        let result = conn.insert_datapoints(&datapoints).await;
        assert!(
            result.is_ok(),
            "Should insert mixed datapoints successfully"
        );
        assert_eq!(
            result.unwrap(),
            4,
            "Should return 4 total written rows (2 chat + 2 JSON)"
        );
    }

    #[tokio::test]
    async fn test_insert_datapoints_with_empty_slice() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        // Expect no calls to run_query_with_external_data when given an empty slice
        mock_clickhouse_client
            .expect_run_query_with_external_data()
            .times(0);

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let datapoints: Vec<DatapointInsert> = vec![];

        let result = conn.insert_datapoints(&datapoints).await;
        assert!(result.is_ok(), "Should handle empty slice successfully");
        assert_eq!(
            result.unwrap(),
            0,
            "Should return 0 written rows for empty slice"
        );
    }

    #[tokio::test]
    async fn test_insert_datapoints_validates_dataset_names() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        // Expect no calls to run_query_with_external_data when validation fails
        mock_clickhouse_client
            .expect_run_query_with_external_data()
            .times(0);

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        // Test with reserved name "builder"
        let datapoints_with_builder = vec![DatapointInsert::Chat(ChatInferenceDatapointInsert {
            dataset_name: "builder".to_string(), // This should fail validation
            function_name: "test_function".to_string(),
            id: Uuid::parse_str("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa").unwrap(),
            name: None,
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![],
            },
            output: None,
            tool_params: None,
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: false,
        })];

        let result = conn.insert_datapoints(&datapoints_with_builder).await;
        assert!(
            result.is_err(),
            "Should fail with reserved dataset name 'builder'"
        );
        let err = result.unwrap_err();
        assert!(
            matches!(err.get_details(), ErrorDetails::InvalidDatasetName { .. }),
            "Should return InvalidDatasetName error"
        );

        // Test with reserved prefix "tensorzero::"
        let datapoints_with_tensorzero_prefix =
            vec![DatapointInsert::Json(JsonInferenceDatapointInsert {
                dataset_name: "tensorzero::reserved".to_string(), // This should fail validation
                function_name: "test_function".to_string(),
                id: Uuid::parse_str("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb").unwrap(),
                name: None,
                episode_id: None,
                input: StoredInput {
                    system: None,
                    messages: vec![],
                },
                output: None,
                output_schema: json!({"type": "object"}),
                tags: None,
                auxiliary: String::new(),
                staled_at: None,
                source_inference_id: None,
                is_custom: false,
            })];

        let result = conn
            .insert_datapoints(&datapoints_with_tensorzero_prefix)
            .await;
        assert!(
            result.is_err(),
            "Should fail with reserved dataset name prefix 'tensorzero::'"
        );
        let err = result.unwrap_err();
        assert!(
            matches!(err.get_details(), ErrorDetails::InvalidDatasetName { .. }),
            "Should return InvalidDatasetName error"
        );
    }

    #[tokio::test]
    async fn test_count_datapoints_for_dataset_function_chat_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(
                    query,
                    "SELECT toUInt32(count()) as count
                    FROM {table:Identifier}
                    WHERE dataset_name = {dataset_name:String}
                    AND function_name = {function_name:String}",
                );

                assert_eq!(parameters.get("table"), Some(&"ChatInferenceDatapoint"));
                assert_eq!(parameters.get("dataset_name"), Some(&"test_dataset"));
                assert_eq!(parameters.get("function_name"), Some(&"test_function"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from("42"),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let params = CountDatapointsForDatasetFunctionParams {
            dataset_name: "test_dataset".to_string(),
            function_name: "test_function".to_string(),
            function_type: DatapointKind::Chat,
        };

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .count_datapoints_for_dataset_function(&params)
            .await
            .unwrap();

        assert_eq!(result, 42, "Should return 42 datapoints");
    }

    #[tokio::test]
    async fn test_count_datapoints_for_dataset_function_json_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(
                    query,
                    "SELECT toUInt32(count()) as count
                    FROM {table:Identifier}
                    WHERE dataset_name = {dataset_name:String}
                    AND function_name = {function_name:String}",
                );

                assert_eq!(parameters.get("table"), Some(&"JsonInferenceDatapoint"));
                assert_eq!(parameters.get("dataset_name"), Some(&"my_dataset"));
                assert_eq!(parameters.get("function_name"), Some(&"my_function"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from("17"),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let params = CountDatapointsForDatasetFunctionParams {
            dataset_name: "my_dataset".to_string(),
            function_name: "my_function".to_string(),
            function_type: DatapointKind::Json,
        };

        let result = conn
            .count_datapoints_for_dataset_function(&params)
            .await
            .unwrap();

        assert_eq!(result, 17, "Should return 17 datapoints");
    }

    #[tokio::test]
    async fn test_get_datapoint_chat_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "WITH dataset as (
                SELECT
                    'chat' as type,
                    dataset_name,
                    function_name,
                    name,
                    id,
                    episode_id,
                    input,
                    output,
                    tool_params,
                    dynamic_tools,
                    dynamic_provider_tools,
                    parallel_tool_calls,
                    tool_choice,
                    allowed_tools,
                    '' as output_schema,");
                assert_query_contains(query,
                    "tags,
                    auxiliary,
                    source_inference_id,
                    is_deleted,
                    is_custom,
                    staled_at,
                    formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at");
                assert_query_contains(query, "FROM ChatInferenceDatapoint AS i FINAL
                    WHERE true
                    AND dataset_name = {dataset_name:String}
                    AND id IN ['123e4567-e89b-12d3-a456-426614174000']
                    AND staled_at IS NULL");
                assert_query_contains(query, "ORDER BY updated_at DESC, id DESC
                    LIMIT {subquery_limit:UInt32}");
                assert_query_contains(query, "UNION ALL");
                assert_query_contains(query, "
                SELECT
                    'json' as type,
                    dataset_name,
                    function_name,
                    name,
                    id,
                    episode_id,
                    input,
                    output,
                    '' as tool_params,");
                assert_query_contains(query,
                "[] as dynamic_tools,
                    [] as dynamic_provider_tools,
                    NULL as parallel_tool_calls,
                    NULL as tool_choice,
                    NULL as allowed_tools,
                    output_schema,
                    tags,
                    auxiliary,
                    source_inference_id,
                    is_deleted,
                    is_custom,
                    staled_at,
                    formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at");
                assert_query_contains(query, "FROM JsonInferenceDatapoint AS i FINAL
                    WHERE true
                    AND dataset_name = {dataset_name:String}
                    AND id IN ['123e4567-e89b-12d3-a456-426614174000']
                    AND staled_at IS NULL");
                assert_query_contains(query, "SELECT *
                    FROM dataset
                    ORDER BY updated_at DESC, id DESC
                    LIMIT {limit:UInt32}
                    OFFSET {offset:UInt32}
                FORMAT JSONEachRow");

                assert_eq!(parameters.get("dataset_name"), Some(&"test_dataset"));
                assert_eq!(parameters.get("limit"), Some(&"1"));
                assert_eq!(parameters.get("offset"), Some(&"0"));
                assert_eq!(parameters.get("subquery_limit"), Some(&"1"));

                assert!(!parameters.contains_key("datapoint_id"), "Datapoint ID should be passed as a list of IDs");

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"type":"chat","dataset_name":"test_dataset","function_name":"test_function","name":"test_name","id":"123e4567-e89b-12d3-a456-426614174000","episode_id":"223e4567-e89b-12d3-a456-426614174000","input":"{\"messages\":[]}","output":"[{\"type\":\"text\",\"text\":\"test output\"}]","tool_params":"{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}","tags":{},"auxiliary":"","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let params = GetDatapointParams {
            dataset_name: "test_dataset".to_string(),
            datapoint_id: Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap(),
            allow_stale: None,
        };

        let result = conn.get_datapoint(&params).await.unwrap();

        if let StoredDatapoint::Chat(datapoint) = result {
            // Verify it's a chat datapoint
            assert_eq!(datapoint.dataset_name, "test_dataset");
            assert_eq!(datapoint.function_name, "test_function");
            assert_eq!(
                datapoint.id.to_string(),
                "123e4567-e89b-12d3-a456-426614174000"
            );
        } else {
            panic!("Expected chat datapoint");
        }
    }

    #[tokio::test]
    async fn test_get_datapoint_json_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "WITH dataset as (");
                assert_query_contains(query, "FROM ChatInferenceDatapoint AS i FINAL");
                assert_query_contains(query, "UNION ALL");
                assert_query_contains(query, "FROM JsonInferenceDatapoint AS i FINAL");
                assert_query_contains(query, "id IN ['323e4567-e89b-12d3-a456-426614174000']");
                assert_query_contains(query, "staled_at IS NULL");

                assert_eq!(parameters.get("dataset_name"), Some(&"json_dataset"));
                assert!(!parameters.contains_key("datapoint_id"), "Datapoint ID should be passed as a list of IDs");

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"type":"json","dataset_name":"json_dataset","function_name":"json_function","name":null,"id":"323e4567-e89b-12d3-a456-426614174000","episode_id":null,"input":"{\"messages\":[]}","output":"{\"parsed\":{\"key\":\"value\"}}","output_schema":"{\"type\":\"object\"}","tags":{},"auxiliary":"","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let params = GetDatapointParams {
            dataset_name: "json_dataset".to_string(),
            datapoint_id: Uuid::parse_str("323e4567-e89b-12d3-a456-426614174000").unwrap(),
            allow_stale: None,
        };

        let result = conn.get_datapoint(&params).await.unwrap();

        if let StoredDatapoint::Json(datapoint) = result {
            // Verify it's a json datapoint
            assert_eq!(datapoint.dataset_name, "json_dataset");
            assert_eq!(datapoint.function_name, "json_function");
            assert_eq!(
                datapoint.id.to_string(),
                "323e4567-e89b-12d3-a456-426614174000"
            );
        } else {
            panic!("Expected json datapoint");
        }
    }

    #[tokio::test]
    async fn test_get_datapoint_returns_not_found() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_does_not_contain(query, "staled_at IS NULL");
                assert_query_contains(query, "id IN ['323e4567-e89b-12d3-a456-426614174000']");

                assert_eq!(parameters.get("dataset_name"), Some(&"json_dataset"));

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

        let params = GetDatapointParams {
            dataset_name: "json_dataset".to_string(),
            datapoint_id: Uuid::parse_str("323e4567-e89b-12d3-a456-426614174000").unwrap(),
            allow_stale: Some(true),
        };

        let error = conn.get_datapoint(&params).await.unwrap_err();
        match error.get_details() {
            ErrorDetails::DatapointNotFound {
                dataset_name,
                datapoint_id,
            } => {
                assert_eq!(dataset_name, "json_dataset");
                assert_eq!(
                    datapoint_id.to_string(),
                    "323e4567-e89b-12d3-a456-426614174000"
                );
            }
            other_details => {
                panic!("Expected DatapointNotFound error, encountered {other_details}");
            }
        }
    }

    #[tokio::test]
    async fn test_get_datapoint_allows_staled() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "id IN ['323e4567-e89b-12d3-a456-426614174000']");
                assert_query_does_not_contain(query, "staled_at IS NULL");

                assert_eq!(parameters.get("dataset_name"), Some(&"json_dataset"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let params = GetDatapointParams {
            dataset_name: "json_dataset".to_string(),
            datapoint_id: Uuid::parse_str("323e4567-e89b-12d3-a456-426614174000").unwrap(),
            allow_stale: Some(true),
        };

        // Don't assert on the result, just checking the queries are correct.
        let _ = conn.get_datapoint(&params).await;
    }

    #[tokio::test]
    async fn test_get_datapoint_not_found() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(), // Empty response means not found
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let params = GetDatapointParams {
            dataset_name: "test_dataset".to_string(),
            datapoint_id: Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap(),
            allow_stale: None,
        };

        let result = conn.get_datapoint(&params).await;

        assert!(
            result.is_err(),
            "Should return error when datapoint not found"
        );
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Datapoint not found"));
    }

    #[tokio::test]
    async fn test_get_datapoints_with_empty_ids() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                // When ids is empty, there should be no ID filter
                assert_query_does_not_contain(query, "id IN");
                assert_query_contains(query, "dataset_name = {dataset_name:String}");
                assert_query_contains(query, "staled_at IS NULL");
                assert_eq!(parameters.get("dataset_name"), Some(&"test_dataset"));
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

        let result = conn
            .get_datapoints(&GetDatapointsParams {
                dataset_name: Some("test_dataset".to_string()),
                function_name: None,
                ids: None,
                limit: u32::MAX,
                offset: 0,
                allow_stale: false,
                filter: None,
                order_by: None,
                search_query_experimental: None,
            })
            .await
            .unwrap();

        assert_eq!(
            result.len(),
            0,
            "Should return empty vector when dataset has no datapoints"
        );
    }

    #[tokio::test]
    async fn test_get_datapoints_with_single_id() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "WITH dataset as (");
                assert_query_contains(query, "FROM ChatInferenceDatapoint AS i FINAL");
                assert_query_contains(query, "UNION ALL");
                assert_query_contains(query, "FROM JsonInferenceDatapoint AS i FINAL");
                assert_query_contains(query, "id IN ['123e4567-e89b-12d3-a456-426614174000']");
                assert_query_contains(query, "staled_at IS NULL");
                assert_query_contains(query, "FORMAT JSONEachRow");

                assert_eq!(parameters.get("dataset_name"), Some(&"test_dataset"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"type":"chat","dataset_name":"test_dataset","function_name":"test_function","name":"test_name","id":"123e4567-e89b-12d3-a456-426614174000","episode_id":"223e4567-e89b-12d3-a456-426614174000","input":"{\"messages\":[]}","output":"[{\"type\":\"text\",\"text\":\"test output\"}]","tool_params":"{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}","tags":{},"auxiliary":"","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let ids = vec![Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()];
        let result = conn
            .get_datapoints(&GetDatapointsParams {
                dataset_name: Some("test_dataset".to_string()),
                function_name: None,
                ids: Some(ids),
                limit: u32::MAX,
                offset: 0,
                allow_stale: false,
                filter: None,
                order_by: None,
                search_query_experimental: None,
            })
            .await
            .unwrap();

        assert_eq!(result.len(), 1, "Should return exactly one datapoint");
        if let StoredDatapoint::Chat(datapoint) = &result[0] {
            assert_eq!(datapoint.dataset_name, "test_dataset");
            assert_eq!(datapoint.function_name, "test_function");
            assert_eq!(
                datapoint.id.to_string(),
                "123e4567-e89b-12d3-a456-426614174000"
            );
        } else {
            panic!("Expected chat datapoint");
        }
    }

    #[tokio::test]
    async fn test_get_datapoints_with_multiple_ids() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "WITH dataset as (");
                assert_query_contains(query, "FROM ChatInferenceDatapoint AS i FINAL");
                assert_query_contains(query, "UNION ALL");
                assert_query_contains(query, "FROM JsonInferenceDatapoint AS i FINAL");
                assert_query_contains(query, "id IN ['123e4567-e89b-12d3-a456-426614174000','223e4567-e89b-12d3-a456-426614174000','323e4567-e89b-12d3-a456-426614174000']");
                assert_query_contains(query, "staled_at IS NULL");

                assert_eq!(parameters.get("dataset_name"), Some(&"test_dataset"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"type":"chat","dataset_name":"test_dataset","function_name":"test_function","name":"test1","id":"123e4567-e89b-12d3-a456-426614174000","episode_id":null,"input":"{\"messages\":[]}","output":"[{\"type\":\"text\",\"text\":\"test output 1\"}]","tool_params":"{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}","tags":{},"auxiliary":"","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z"}
{"type":"chat","dataset_name":"test_dataset","function_name":"test_function","name":"test2","id":"223e4567-e89b-12d3-a456-426614174000","episode_id":null,"input":"{\"messages\":[]}","output":"[{\"type\":\"text\",\"text\":\"test output 2\"}]","tool_params":"{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}","tags":{},"auxiliary":"","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z"}
{"type":"chat","dataset_name":"test_dataset","function_name":"test_function","name":"test3","id":"323e4567-e89b-12d3-a456-426614174000","episode_id":null,"input":"{\"messages\":[]}","output":"[{\"type\":\"text\",\"text\":\"test output 3\"}]","tool_params":"{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}","tags":{},"auxiliary":"","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 3,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let ids = vec![
            Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap(),
            Uuid::parse_str("223e4567-e89b-12d3-a456-426614174000").unwrap(),
            Uuid::parse_str("323e4567-e89b-12d3-a456-426614174000").unwrap(),
        ];
        let result = conn
            .get_datapoints(&GetDatapointsParams {
                dataset_name: Some("test_dataset".to_string()),
                function_name: None,
                ids: Some(ids),
                limit: u32::MAX,
                offset: 0,
                allow_stale: false,
                filter: None,
                order_by: None,
                search_query_experimental: None,
            })
            .await
            .unwrap();

        assert_eq!(result.len(), 3, "Should return three datapoints");

        // Verify all datapoints are returned with correct IDs
        let returned_ids: Vec<String> = result.iter().map(|dp| dp.id().to_string()).collect();
        assert!(returned_ids.contains(&"123e4567-e89b-12d3-a456-426614174000".to_string()));
        assert!(returned_ids.contains(&"223e4567-e89b-12d3-a456-426614174000".to_string()));
        assert!(returned_ids.contains(&"323e4567-e89b-12d3-a456-426614174000".to_string()));
    }

    #[tokio::test]
    async fn test_get_datapoints_respects_allow_stale_false() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "staled_at IS NULL");
                assert_eq!(parameters.get("dataset_name"), Some(&"test_dataset"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(), // Empty response
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let ids = vec![Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()];
        let result = conn
            .get_datapoints(&GetDatapointsParams {
                dataset_name: Some("test_dataset".to_string()),
                function_name: None,
                ids: Some(ids),
                limit: u32::MAX,
                offset: 0,
                allow_stale: false,
                filter: None,
                order_by: None,
                search_query_experimental: None,
            })
            .await
            .unwrap();

        assert_eq!(
            result.len(),
            0,
            "Should not return staled datapoints when allow_stale=false"
        );
    }

    #[tokio::test]
    async fn test_get_datapoints_respects_allow_stale_true() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_does_not_contain(query, "staled_at IS NULL");
                assert_eq!(parameters.get("dataset_name"), Some(&"test_dataset"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"type":"chat","dataset_name":"test_dataset","function_name":"test_function","name":"test_name","id":"123e4567-e89b-12d3-a456-426614174000","episode_id":null,"input":"{\"messages\":[]}","output":"[{\"type\":\"text\",\"text\":\"test output\"}]","tool_params":"{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}","tags":{},"auxiliary":"","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":"2023-01-01T00:00:00Z","updated_at":"2023-01-01T00:00:00Z"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let ids = vec![Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()];
        let result = conn
            .get_datapoints(&GetDatapointsParams {
                dataset_name: Some("test_dataset".to_string()),
                function_name: None,
                ids: Some(ids),
                limit: u32::MAX,
                offset: 0,
                allow_stale: true,
                filter: None,
                order_by: None,
                search_query_experimental: None,
            })
            .await
            .unwrap();

        assert_eq!(
            result.len(),
            1,
            "Should return staled datapoints when allow_stale=true"
        );
        if let StoredDatapoint::Chat(datapoint) = &result[0] {
            assert!(
                datapoint.staled_at.is_some(),
                "Datapoint should have staled_at timestamp"
            );
        } else {
            panic!("Expected chat datapoint");
        }
    }

    #[tokio::test]
    async fn test_get_datapoints_with_mixed_chat_and_json_results() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "FROM ChatInferenceDatapoint AS i FINAL");
                assert_query_contains(query, "UNION ALL");
                assert_query_contains(query, "FROM JsonInferenceDatapoint AS i FINAL");
                assert_query_contains(query, "id IN ['123e4567-e89b-12d3-a456-426614174000','223e4567-e89b-12d3-a456-426614174000']");

                assert_eq!(parameters.get("dataset_name"), Some(&"test_dataset"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"type":"chat","dataset_name":"test_dataset","function_name":"chat_function","name":"chat1","id":"123e4567-e89b-12d3-a456-426614174000","episode_id":null,"input":"{\"messages\":[]}","output":"[{\"type\":\"text\",\"text\":\"chat output\"}]","tool_params":"{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}","tags":{},"auxiliary":"","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z"}
{"type":"json","dataset_name":"test_dataset","function_name":"json_function","name":"json1","id":"223e4567-e89b-12d3-a456-426614174000","episode_id":null,"input":"{\"messages\":[]}","output":"{\"parsed\":{\"key\":\"value\"}}","output_schema":"{\"type\":\"object\"}","tags":{},"auxiliary":"","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 2,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let ids = vec![
            Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap(),
            Uuid::parse_str("223e4567-e89b-12d3-a456-426614174000").unwrap(),
        ];
        let result = conn
            .get_datapoints(&GetDatapointsParams {
                dataset_name: Some("test_dataset".to_string()),
                function_name: None,
                ids: Some(ids),
                limit: u32::MAX,
                offset: 0,
                allow_stale: false,
                filter: None,
                order_by: None,
                search_query_experimental: None,
            })
            .await
            .unwrap();

        assert_eq!(result.len(), 2, "Should return two datapoints");

        // Count types
        let chat_count = result
            .iter()
            .filter(|dp| matches!(dp, StoredDatapoint::Chat(_)))
            .count();
        let json_count = result
            .iter()
            .filter(|dp| matches!(dp, StoredDatapoint::Json(_)))
            .count();

        assert_eq!(chat_count, 1, "Should have 1 chat datapoint");
        assert_eq!(json_count, 1, "Should have 1 json datapoint");

        // Verify chat datapoint
        if let StoredDatapoint::Chat(chat_dp) = result
            .iter()
            .find(|dp| matches!(dp, StoredDatapoint::Chat(_)))
            .unwrap()
        {
            assert_eq!(chat_dp.function_name, "chat_function");
            assert_eq!(chat_dp.name, Some("chat1".to_string()));
        }

        // Verify json datapoint
        if let StoredDatapoint::Json(json_dp) = result
            .iter()
            .find(|dp| matches!(dp, StoredDatapoint::Json(_)))
            .unwrap()
        {
            assert_eq!(json_dp.function_name, "json_function");
            assert_eq!(json_dp.name, Some("json1".to_string()));
        }
    }

    #[tokio::test]
    async fn test_get_datapoints_with_empty_response() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, _| {
                assert_query_contains(query, "WITH dataset as (");
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(), // Empty response
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let ids = vec![Uuid::parse_str("999e4567-e89b-12d3-a456-426614174000").unwrap()];
        let result = conn
            .get_datapoints(&GetDatapointsParams {
                dataset_name: Some("test_dataset".to_string()),
                function_name: None,
                ids: Some(ids),
                limit: u32::MAX,
                offset: 0,
                allow_stale: false,
                filter: None,
                order_by: None,
                search_query_experimental: None,
            })
            .await
            .unwrap();

        assert_eq!(
            result.len(),
            0,
            "Should return empty vector for non-existent IDs"
        );
    }

    #[tokio::test]
    async fn test_get_datapoints_with_function_name_filter() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "AND function_name = {function_name:String}");

                assert_eq!(parameters.get("function_name"), Some(&"test_function"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"type":"chat","dataset_name":"test_dataset","function_name":"test_function","name":"test_name","id":"123e4567-e89b-12d3-a456-426614174000","episode_id":"223e4567-e89b-12d3-a456-426614174000","input":"{\"messages\":[]}","output":"[{\"type\":\"text\",\"text\":\"test output\"}]","tool_params":"{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}","tags":{},"auxiliary":"","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .get_datapoints(&GetDatapointsParams {
                dataset_name: Some("test_dataset".to_string()),
                function_name: Some("test_function".to_string()),
                ids: None,
                limit: 20,
                offset: 0,
                allow_stale: false,
                filter: None,
                order_by: None,
                search_query_experimental: None,
            })
            .await
            .unwrap();

        assert_eq!(result.len(), 1, "Should return one datapoint");
    }

    #[tokio::test]
    async fn test_get_datapoints_with_tag_filter() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "AND i.tags[{p0:String}] = {p1:String}");

                assert_eq!(parameters.get("p0"), Some(&"environment"));
                assert_eq!(parameters.get("p1"), Some(&"production"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"type":"chat","dataset_name":"test_dataset","function_name":"test_function","name":"test_name","id":"123e4567-e89b-12d3-a456-426614174000","episode_id":"223e4567-e89b-12d3-a456-426614174000","input":"{\"messages\":[]}","output":"[{\"type\":\"text\",\"text\":\"test output\"}]","tool_params":"{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}","tags":{"environment":"production"},"auxiliary":"","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let filter = DatapointFilter::Tag(TagFilter {
            key: "environment".to_string(),
            value: "production".to_string(),
            comparison_operator: TagComparisonOperator::Equal,
        });

        let result = conn
            .get_datapoints(&GetDatapointsParams {
                dataset_name: Some("test_dataset".to_string()),
                function_name: None,
                ids: None,
                limit: 20,
                offset: 0,
                allow_stale: false,
                filter: Some(filter),
                order_by: None,
                search_query_experimental: None,
            })
            .await
            .unwrap();

        assert_eq!(result.len(), 1, "Should return one datapoint");
    }

    #[tokio::test]
    async fn test_get_datapoints_with_search_query() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, _parameters| {
                // Should include term frequency calculations
                assert_query_contains(query, "ifNull(countSubstringsCaseInsensitiveUTF8(input,");
                assert_query_contains(query, "ifNull(countSubstringsCaseInsensitiveUTF8(output,");
                assert_query_contains(
                    query,
                    "input_term_frequency + output_term_frequency as total_term_frequency",
                );
                // Should filter by total_term_frequency > 0
                assert_query_contains(query, "AND total_term_frequency > 0");
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"type":"chat","dataset_name":"test_dataset","function_name":"test_function","name":"test_name","id":"123e4567-e89b-12d3-a456-426614174000","episode_id":null,"input":"{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hello\"}]}]}","output":"[{\"type\":\"text\",\"text\":\"hello world\"}]","tool_params":"{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}","tags":{},"auxiliary":"","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z","input_term_frequency":1,"output_term_frequency":1,"total_term_frequency":2}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .get_datapoints(&GetDatapointsParams {
                dataset_name: Some("test_dataset".to_string()),
                function_name: None,
                ids: None,
                limit: 20,
                offset: 0,
                allow_stale: false,
                filter: None,
                order_by: None,
                search_query_experimental: Some("hello".to_string()),
            })
            .await
            .unwrap();

        assert_eq!(result.len(), 1, "Should return one datapoint");
    }

    #[tokio::test]
    async fn test_get_datapoints_with_order_by_timestamp_desc() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, _parameters| {
                // Should order by timestamp in descending order
                assert_query_contains(query, "ORDER BY updated_at DESC");
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"type":"chat","dataset_name":"test_dataset","function_name":"test_function","name":"test_name","id":"123e4567-e89b-12d3-a456-426614174000","episode_id":null,"input":"{\"messages\":[]}","output":"[{\"type\":\"text\",\"text\":\"test output\"}]","tool_params":"{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}","tags":{},"auxiliary":"","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z","input_term_frequency":1,"output_term_frequency":1,"total_term_frequency":2}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .get_datapoints(&GetDatapointsParams {
                dataset_name: Some("test_dataset".to_string()),
                function_name: None,
                ids: None,
                limit: 20,
                offset: 0,
                allow_stale: false,
                filter: None,
                order_by: Some(vec![DatapointOrderBy {
                    term: DatapointOrderByTerm::Timestamp,
                    direction: OrderDirection::Desc,
                }]),
                search_query_experimental: None,
            })
            .await
            .unwrap();

        assert_eq!(result.len(), 1, "Should return one datapoint");
    }

    #[tokio::test]
    async fn test_get_datapoints_with_order_by_timestamp_asc() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, _parameters| {
                // Should order by timestamp in ascending order
                assert_query_contains(query, "ORDER BY updated_at ASC");
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"type":"chat","dataset_name":"test_dataset","function_name":"test_function","name":"test_name","id":"123e4567-e89b-12d3-a456-426614174000","episode_id":null,"input":"{\"messages\":[]}","output":"[{\"type\":\"text\",\"text\":\"test output\"}]","tool_params":"{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}","tags":{},"auxiliary":"","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z","input_term_frequency":1,"output_term_frequency":1,"total_term_frequency":2}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .get_datapoints(&GetDatapointsParams {
                dataset_name: Some("test_dataset".to_string()),
                function_name: None,
                ids: None,
                limit: 20,
                offset: 0,
                allow_stale: false,
                filter: None,
                order_by: Some(vec![DatapointOrderBy {
                    term: DatapointOrderByTerm::Timestamp,
                    direction: OrderDirection::Asc,
                }]),
                search_query_experimental: None,
            })
            .await
            .unwrap();

        assert_eq!(result.len(), 1, "Should return one datapoint");
    }

    #[tokio::test]
    async fn test_get_datapoints_with_order_by_search_relevance() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, _parameters| {
                // Should include term frequency calculations
                assert_query_contains(query, "ifNull(countSubstringsCaseInsensitiveUTF8(input,");
                assert_query_contains(query, "ifNull(countSubstringsCaseInsensitiveUTF8(output,");
                // Should order by total_term_frequency in descending order
                assert_query_contains(query, "ORDER BY total_term_frequency DESC");
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"type":"chat","dataset_name":"test_dataset","function_name":"test_function","name":"test_name","id":"123e4567-e89b-12d3-a456-426614174000","episode_id":null,"input":"{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hello\"}]}]}","output":"[{\"type\":\"text\",\"text\":\"hello world\"}]","tool_params":"{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}","tags":{},"auxiliary":"","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z","input_term_frequency":1,"output_term_frequency":1,"total_term_frequency":2}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .get_datapoints(&GetDatapointsParams {
                dataset_name: Some("test_dataset".to_string()),
                function_name: None,
                ids: None,
                limit: 20,
                offset: 0,
                allow_stale: false,
                filter: None,
                order_by: Some(vec![DatapointOrderBy {
                    term: DatapointOrderByTerm::SearchRelevance,
                    direction: OrderDirection::Desc,
                }]),
                search_query_experimental: Some("hello".to_string()),
            })
            .await
            .unwrap();

        assert_eq!(result.len(), 1, "Should return one datapoint");
    }

    #[tokio::test]
    async fn test_get_datapoints_with_multiple_order_by() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, _parameters| {
                // Should order by search relevance first, then by timestamp
                assert_query_contains(query, "ORDER BY total_term_frequency DESC, updated_at ASC");
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"type":"chat","dataset_name":"test_dataset","function_name":"test_function","name":"test_name","id":"123e4567-e89b-12d3-a456-426614174000","episode_id":null,"input":"{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hello\"}]}]}","output":"[{\"type\":\"text\",\"text\":\"hello world\"}]","tool_params":"{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}","tags":{},"auxiliary":"","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z","input_term_frequency":1,"output_term_frequency":1,"total_term_frequency":2}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .get_datapoints(&GetDatapointsParams {
                dataset_name: Some("test_dataset".to_string()),
                function_name: None,
                ids: None,
                limit: 20,
                offset: 0,
                allow_stale: false,
                filter: None,
                order_by: Some(vec![
                    DatapointOrderBy {
                        term: DatapointOrderByTerm::SearchRelevance,
                        direction: OrderDirection::Desc,
                    },
                    DatapointOrderBy {
                        term: DatapointOrderByTerm::Timestamp,
                        direction: OrderDirection::Asc,
                    },
                ]),
                search_query_experimental: Some("hello".to_string()),
            })
            .await
            .unwrap();

        assert_eq!(result.len(), 1, "Should return one datapoint");
    }

    #[tokio::test]
    async fn test_get_datapoints_search_relevance_without_search_query_fails() {
        let mock_clickhouse_client = MockClickHouseClient::new();
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .get_datapoints(&GetDatapointsParams {
                dataset_name: Some("test_dataset".to_string()),
                function_name: None,
                ids: None,
                limit: 20,
                offset: 0,
                allow_stale: false,
                filter: None,
                order_by: Some(vec![DatapointOrderBy {
                    term: DatapointOrderByTerm::SearchRelevance,
                    direction: OrderDirection::Desc,
                }]),
                search_query_experimental: None,
            })
            .await;

        assert!(
            result.is_err(),
            "Should fail when using SearchRelevance without search query"
        );
        let err = result.unwrap_err();
        assert!(err.to_string().contains("search_query_experimental"));
    }

    #[tokio::test]
    async fn test_delete_datapoints_with_specific_ids() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();

        // Expect two queries: one for ChatInferenceDatapoint, one for JsonInferenceDatapoint
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .times(2)
            .withf(move |query, parameters| {
                assert_query_contains(query, "INSERT INTO");
                assert_query_contains(
                    query,
                    "SELECT * REPLACE ( now64() AS updated_at, now64() AS staled_at )",
                );
                assert_query_contains(query, "WHERE dataset_name = {dataset_name:String}");
                assert_query_contains(query, &format!("AND id IN ['{id1}','{id2}']"));

                assert_eq!(parameters.get("dataset_name"), Some(&"test_dataset"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 2,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .delete_datapoints("test_dataset", Some(&[id1, id2]))
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 4); // 2 from chat + 2 from json
    }

    #[tokio::test]
    async fn test_delete_datapoints_with_empty_ids_deletes_all() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        // Expect two queries: one for ChatInferenceDatapoint, one for JsonInferenceDatapoint
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .times(2)
            .withf(|query, parameters| {
                // Verify the query structure
                assert_query_contains(query, "INSERT INTO");
                assert_query_contains(
                    query,
                    "SELECT * REPLACE ( now64() AS updated_at, now64() AS staled_at )",
                );
                assert_query_contains(query, "WHERE dataset_name = {dataset_name:String}");

                assert_query_does_not_contain(query, "AND id IN");

                assert_eq!(parameters.get("dataset_name"), Some(&"test_dataset"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 5,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn.delete_datapoints("test_dataset", None).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 10); // 5 from chat + 5 from json
    }

    #[tokio::test]
    async fn test_delete_datapoints_queries_both_tables() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let id1 = Uuid::now_v7();

        // The queries are executed in parallel via try_join!, so we can't rely on ordering
        // Just verify that both table types are queried
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .times(2)
            .withf(move |query, _parameters| {
                query.contains("ChatInferenceDatapoint") || query.contains("JsonInferenceDatapoint")
            })
            .returning(|query, _| {
                // Return different row counts for each table to verify aggregation
                if query.contains("ChatInferenceDatapoint") {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 1,
                        },
                    })
                } else if query.contains("JsonInferenceDatapoint") {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 2,
                        },
                    })
                } else {
                    panic!("Unexpected query: {query}");
                }
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn.delete_datapoints("my_dataset", Some(&[id1])).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 3); // 1 from chat + 2 from json
    }
}
