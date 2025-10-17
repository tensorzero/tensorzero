use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::num::ParseIntError;
use uuid::Uuid;

use crate::config::{MetricConfigLevel, MetricConfigType};
use crate::db::clickhouse::query_builder::FloatComparisonOperator;
use crate::db::clickhouse::{ClickHouseConnectionInfo, Rows, TableName};
// TODO: move things somewhere sensible
use crate::endpoints::datasets::{validate_dataset_name, Datapoint, DatapointKind};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::{ContentBlockChatOutput, JsonInferenceOutput, StoredInput};
use crate::serde_util::{
    deserialize_optional_string_or_parsed_json, deserialize_string_or_parsed_json,
};
use crate::tool::ToolCallConfigDatabaseInsert;

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum DatasetOutputSource {
    // When generating a dataset, don't include any output.
    None,
    // When generating a dataset, include original inference output.
    Inference,
    // When generating a dataset, include any linked demonstration as output.
    Demonstration,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct MetricFilter {
    pub metric: String,
    pub metric_type: MetricConfigType,
    pub operator: FloatComparisonOperator,
    pub threshold: f64,
    pub join_on: MetricConfigLevel,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub enum DatapointInsert {
    #[serde(rename = "chat")]
    Chat(ChatInferenceDatapointInsert),
    #[serde(rename = "json")]
    Json(JsonInferenceDatapointInsert),
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct ChatInferenceDatapointInsert {
    pub dataset_name: String,
    pub function_name: String,
    pub name: Option<String>,
    pub id: Uuid,
    pub episode_id: Option<Uuid>,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: StoredInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_optional_string_or_parsed_json")]
    pub output: Option<Vec<ContentBlockChatOutput>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_optional_string_or_parsed_json")]
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<HashMap<String, String>>,
    pub auxiliary: String,
    pub staled_at: Option<String>,
    pub source_inference_id: Option<Uuid>,
    pub is_custom: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct JsonInferenceDatapointInsert {
    pub dataset_name: String,
    pub function_name: String,
    pub name: Option<String>,
    pub id: Uuid,
    pub episode_id: Option<Uuid>,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: StoredInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_optional_string_or_parsed_json")]
    pub output: Option<JsonInferenceOutput>,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub output_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<HashMap<String, String>>,
    pub auxiliary: String,
    pub staled_at: Option<String>,
    pub source_inference_id: Option<Uuid>,
    pub is_custom: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct DatasetQueryParams {
    pub inference_type: DatapointKind,
    pub function_name: Option<String>,
    pub dataset_name: Option<String>,
    pub variant_name: Option<String>,
    pub extra_where: Option<Vec<String>>,
    pub extra_params: Option<HashMap<String, String>>,
    // TODO: consider supporting compound filters (e.g. AND/OR)
    pub metric_filter: Option<MetricFilter>,
    pub output_source: DatasetOutputSource,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct GetDatasetRowsParams {
    pub dataset_name: String,
    pub page_size: u32,
    pub offset: u32,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct GetDatasetMetadataParams {
    // Only select datasets matching a specific function
    pub function_name: Option<String>,
    pub page_size: Option<u32>,
    pub offset: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct DatasetDetailRow {
    pub id: String,
    #[serde(rename = "type")]
    pub row_type: String,
    pub function_name: String,
    pub name: Option<String>,
    pub episode_id: Option<String>,
    pub updated_at: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct DatasetMetadata {
    pub dataset_name: String,
    pub count: u32,
    pub last_updated: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct AdjacentDatapointIds {
    pub previous_id: Option<Uuid>,
    pub next_id: Option<Uuid>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct StaleDatapointParams {
    pub dataset_name: String,
    pub datapoint_id: Uuid,
    pub function_type: DatapointKind,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct CountDatapointsForDatasetFunctionParams {
    pub dataset_name: String,
    pub function_name: String,
    pub function_type: DatapointKind,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct GetAdjacentDatapointIdsParams {
    pub dataset_name: String,
    pub datapoint_id: Uuid,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export, optional_fields))]
pub struct GetDatapointParams {
    pub dataset_name: String,
    pub datapoint_id: Uuid,
    /// Whether to include stale datapoints in the query; false by default.
    pub allow_stale: Option<bool>,
}

#[async_trait]
pub trait DatasetQueries {
    /// Counts rows for a dataset based on query parameters
    async fn count_rows_for_dataset(&self, params: &DatasetQueryParams) -> Result<u32, Error>;

    /// Inserts rows into a dataset table by selecting from the inference tables
    /// Returns the number of rows inserted
    async fn insert_rows_for_dataset(&self, params: &DatasetQueryParams) -> Result<u32, Error>;

    /// Gets rows from a dataset with pagination
    async fn get_dataset_rows(
        &self,
        params: &GetDatasetRowsParams,
    ) -> Result<Vec<DatasetDetailRow>, Error>;

    /// Gets dataset metadata (name, count, last updated)
    async fn get_dataset_metadata(
        &self,
        params: &GetDatasetMetadataParams,
    ) -> Result<Vec<DatasetMetadata>, Error>;

    /// Gets the count of unique dataset names
    async fn count_datasets(&self) -> Result<u32, Error>;

    /// Marks a datapoint as stale by inserting a new row with staled_at set to now
    async fn stale_datapoint(&self, params: &StaleDatapointParams) -> Result<(), Error>;

    /// Inserts a new datapoint into the dataset
    async fn insert_datapoint(&self, datapoint: &DatapointInsert) -> Result<(), Error>;

    /// Counts datapoints for a specific dataset and function
    async fn count_datapoints_for_dataset_function(
        &self,
        params: &CountDatapointsForDatasetFunctionParams,
    ) -> Result<u32, Error>;

    /// Gets the adjacent (previous and next) datapoint IDs for a given datapoint
    async fn get_adjacent_datapoint_ids(
        &self,
        params: &GetAdjacentDatapointIdsParams,
    ) -> Result<AdjacentDatapointIds, Error>;

    /// Gets a single datapoint by dataset name and ID
    async fn get_datapoint(&self, params: &GetDatapointParams) -> Result<Datapoint, Error>;
}

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
        let type_specific_field = match params.inference_type {
            DatapointKind::Chat => "subquery.tool_params",
            DatapointKind::Json => "subquery.output_schema",
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
                {type_specific_field},
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

    async fn get_dataset_rows(
        &self,
        params: &GetDatasetRowsParams,
    ) -> Result<Vec<DatasetDetailRow>, Error> {
        let dataset_name = &params.dataset_name;
        let page_size = params.page_size;

        // Ensure offset is not negative
        let offset = std::cmp::max(0, params.offset);

        let query = r"
            SELECT *
            FROM (
                SELECT
                    id,
                    'chat' as type,
                    function_name,
                    name,
                    episode_id,
                    formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at
                FROM ChatInferenceDatapoint
                FINAL
                WHERE dataset_name = {dataset_name:String} AND staled_at IS NULL
                UNION ALL
                SELECT
                    id,
                    'json' as type,
                    function_name,
                    name,
                    episode_id,
                    formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at
                FROM JsonInferenceDatapoint
                FINAL
                WHERE dataset_name = {dataset_name:String} AND staled_at IS NULL
            )
            ORDER BY updated_at DESC, id DESC
            LIMIT {page_size:UInt32}
            OFFSET {offset:UInt32}
            FORMAT JSONEachRow
        ";

        let page_size_str = page_size.to_string();
        let offset_str = offset.to_string();

        let mut query_params = HashMap::new();
        query_params.insert("dataset_name", dataset_name.as_str());
        query_params.insert("page_size", page_size_str.as_str());
        query_params.insert("offset", offset_str.as_str());

        let response = self
            .run_query_synchronous(query.to_string(), &query_params)
            .await?;

        // Parse the response as JSON lines
        let rows: Vec<DatasetDetailRow> = response
            .response
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                serde_json::from_str(line).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: format!("Failed to deserialize DatasetDetailRow: {e}"),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(rows)
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
        let limit_clause = match params.page_size {
            Some(page_size) => {
                query_params_owned.insert("page_size".to_string(), page_size.to_string());
                "LIMIT {page_size:UInt32}"
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

    async fn stale_datapoint(&self, params: &StaleDatapointParams) -> Result<(), Error> {
        let StaleDatapointParams {
            dataset_name,
            datapoint_id,
            function_type,
        } = params;

        let table = function_type.table_name();

        let type_specific_field = match function_type {
            DatapointKind::Chat => "tool_params",
            DatapointKind::Json => "output_schema",
        };

        let query = format!(
            r"
            INSERT INTO {{table:Identifier}}
            (
                dataset_name,
                function_name,
                id,
                name,
                episode_id,
                input,
                output,
                {type_specific_field},
                tags,
                auxiliary,
                is_deleted,
                source_inference_id,
                is_custom,
                staled_at,
                updated_at
            )
            SELECT
                dataset_name,
                function_name,
                id,
                name,
                episode_id,
                input,
                output,
                {type_specific_field},
                tags,
                auxiliary,
                is_deleted,
                source_inference_id,
                is_custom,
                now64() as staled_at,
                now64() as updated_at
            FROM {{table:Identifier}} FINAL
            WHERE dataset_name = {{dataset_name:String}} AND id = {{datapoint_id:String}}
            "
        );

        let datapoint_id_str = datapoint_id.to_string();
        let mut query_params = HashMap::new();
        query_params.insert("table", table.as_str());
        query_params.insert("dataset_name", dataset_name.as_str());
        query_params.insert("datapoint_id", datapoint_id_str.as_str());

        self.run_query_synchronous(query, &query_params).await?;

        Ok(())
    }

    async fn insert_datapoint(&self, datapoint: &DatapointInsert) -> Result<(), Error> {
        match datapoint {
            DatapointInsert::Chat(chat_datapoint) => {
                validate_dataset_name(&chat_datapoint.dataset_name)?;

                // Build the struct for the insert; to match what ClickHouse expects, these values are either JSON objects or an empty string (in the case of null).
                // tool_params in clickhouse is a non-null String
                let tool_params_value = if let Some(tool_params) = &chat_datapoint.tool_params {
                    serde_json::to_value(tool_params)?
                } else {
                    serde_json::json!("")
                };
                // Tags in clickhouse is a Non-null Map(String, String)
                let tags_value = if let Some(tags) = &chat_datapoint.tags {
                    serde_json::to_value(tags)?
                } else {
                    serde_json::json!({})
                };
                let value = serde_json::json!({
                    "dataset_name": chat_datapoint.dataset_name,
                    "function_name": chat_datapoint.function_name,
                    "id": chat_datapoint.id,
                    "name": chat_datapoint.name,
                    "episode_id": chat_datapoint.episode_id,
                    "input": chat_datapoint.input,
                    "output": chat_datapoint.output,
                    "tool_params": tool_params_value,
                    "tags": tags_value,
                    "auxiliary": chat_datapoint.auxiliary,
                    "source_inference_id": chat_datapoint.source_inference_id,
                    "is_custom": chat_datapoint.is_custom,
                });

                // Serialize to JSON string
                let value_str = serde_json::to_string(&value).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: e.to_string(),
                    })
                })?;

                self.write_non_batched::<()>(
                    Rows::Serialized(&[value_str]),
                    TableName::ChatInferenceDatapoint,
                )
                .await?;
            }
            DatapointInsert::Json(json_datapoint) => {
                validate_dataset_name(&json_datapoint.dataset_name)?;

                // Tags in clickhouse is a Non-null Map(String, String)
                let tags_value = if let Some(tags) = &json_datapoint.tags {
                    serde_json::to_value(tags)?
                } else {
                    serde_json::json!({})
                };
                let value = serde_json::json!({
                    "dataset_name": json_datapoint.dataset_name,
                    "function_name": json_datapoint.function_name,
                    "id": json_datapoint.id,
                    "name": json_datapoint.name,
                    "episode_id": json_datapoint.episode_id,
                    "input": json_datapoint.input,
                    "output": json_datapoint.output,
                    "output_schema": json_datapoint.output_schema,
                    "tags": tags_value,
                    "auxiliary": json_datapoint.auxiliary,
                    "source_inference_id": json_datapoint.source_inference_id,
                    "is_custom": json_datapoint.is_custom,
                });

                // Serialize to JSON string
                let value_str = serde_json::to_string(&value).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: e.to_string(),
                    })
                })?;

                self.write_non_batched::<()>(
                    Rows::Serialized(&[value_str]),
                    TableName::JsonInferenceDatapoint,
                )
                .await?;
            }
        }

        Ok(())
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

    async fn get_adjacent_datapoint_ids(
        &self,
        params: &GetAdjacentDatapointIdsParams,
    ) -> Result<AdjacentDatapointIds, Error> {
        let query = r"
            WITH DatasetIds AS (
                SELECT toUInt128(id) as id_uint FROM ChatInferenceDatapoint WHERE dataset_name = {dataset_name:String}
                UNION ALL
                SELECT toUInt128(id) as id_uint FROM JsonInferenceDatapoint WHERE dataset_name = {dataset_name:String}
            )
            SELECT
                NULLIF(
                    (SELECT uint_to_uuid(min(id_uint)) FROM DatasetIds WHERE id_uint > toUInt128({datapoint_id:UUID})),
                    toUUID('00000000-0000-0000-0000-000000000000')
                ) as next_id,
                NULLIF(
                    (SELECT uint_to_uuid(max(id_uint)) FROM DatasetIds WHERE id_uint < toUInt128({datapoint_id:UUID})),
                    toUUID('00000000-0000-0000-0000-000000000000')
                ) as previous_id
            FORMAT JSONEachRow
        ";

        let datapoint_id_str = params.datapoint_id.to_string();
        let mut query_params = HashMap::new();
        query_params.insert("dataset_name", params.dataset_name.as_str());
        query_params.insert("datapoint_id", datapoint_id_str.as_str());

        let response = self
            .run_query_synchronous(query.to_string(), &query_params)
            .await?;

        let result: AdjacentDatapointIds =
            serde_json::from_str(response.response.trim()).map_err(|e| {
                Error::new(ErrorDetails::ClickHouseDeserialization {
                    message: format!("Failed to deserialize AdjacentDatapointIds: {e}"),
                })
            })?;

        Ok(result)
    }

    async fn get_datapoint(&self, params: &GetDatapointParams) -> Result<Datapoint, Error> {
        const DEFAULT_ALLOW_STALE_IN_GET_DATAPOINT: bool = false;
        let allow_stale = params
            .allow_stale
            .unwrap_or(DEFAULT_ALLOW_STALE_IN_GET_DATAPOINT);

        let allow_stale_clause = if allow_stale {
            // If we allow staled datapoints, we don't need to filter by staled_at.
            ""
        } else {
            "AND staled_at IS NULL"
        };

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
                '' as output_schema, -- for column alignment in UNION ALL
                tags,
                auxiliary,
                source_inference_id,
                is_deleted,
                is_custom,
                staled_at,
                formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at
            FROM ChatInferenceDatapoint FINAL
            WHERE dataset_name = {{dataset_name:String}}
                AND id = {{datapoint_id:String}}
                {allow_stale_clause}
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
                output_schema,
                tags,
                auxiliary,
                source_inference_id,
                is_deleted,
                is_custom,
                staled_at,
                formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at
            FROM JsonInferenceDatapoint FINAL
            WHERE dataset_name = {{dataset_name:String}}
                AND id = {{datapoint_id:String}}
                {allow_stale_clause}
        )
        SELECT * FROM dataset
        LIMIT 1
        FORMAT JSONEachRow
        "
        );

        let datapoint_id = params.datapoint_id.to_string();
        let query_params = HashMap::from([
            ("dataset_name", params.dataset_name.as_str()),
            ("datapoint_id", datapoint_id.as_str()),
        ]);

        let result = self
            .run_query_synchronous(query.to_string(), &query_params)
            .await?;

        if result.response.is_empty() {
            return Err(Error::new(ErrorDetails::DatapointNotFound {
                dataset_name: params.dataset_name.clone(),
                datapoint_id: params.datapoint_id,
            }));
        }

        let datapoint: Datapoint = serde_json::from_str(result.response.trim()).map_err(|e| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: format!("Failed to deserialize datapoint: {e}"),
            })
        })?;

        Ok(datapoint)
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
    let type_specific_field = match params.inference_type {
        DatapointKind::Chat => "tool_params",
        DatapointKind::Json => "output_schema",
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
            {type_specific_field},
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
    use std::sync::Arc;

    use serde_json::json;

    use crate::db::clickhouse::clickhouse_client::MockClickHouseClient;
    use crate::db::clickhouse::ClickHouseResponse;
    use crate::db::clickhouse::ClickHouseResponseMetadata;
    use crate::inference::types::Text;

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

    /// Normalize whitespace and newlines in a query for comparison
    fn normalize_whitespace(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Assert that the query contains a section (ignoring whitespace and newline differences)
    fn assert_query_contains(query: &str, expected_section: &str) {
        let normalized_query = normalize_whitespace(query);
        let normalized_section = normalize_whitespace(expected_section);
        assert!(
            normalized_query.contains(&normalized_section),
            "Query does not contain expected section.\nExpected section: {normalized_section}\nFull query: {normalized_query}"
        );
    }

    fn assert_query_does_not_contain(query: &str, unexpected_section: &str) {
        let normalized_query = normalize_whitespace(query);
        let normalized_section = normalize_whitespace(unexpected_section);
        assert!(
            !normalized_query.contains(&normalized_section),
            "Query contains unexpected section: {normalized_section}\nFull query: {normalized_query}"
        );
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
    async fn test_get_dataset_rows_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "
                SELECT *
                FROM (
                    SELECT
                        id,
                        'chat' as type,
                        function_name,
                        name,
                        episode_id,
                        formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at
                    FROM ChatInferenceDatapoint
                    FINAL
                    WHERE dataset_name = {dataset_name:String} AND staled_at IS NULL
                    UNION ALL
                    SELECT
                        id,
                        'json' as type,
                        function_name,
                        name,
                        episode_id,
                        formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at
                    FROM JsonInferenceDatapoint
                    FINAL
                    WHERE dataset_name = {dataset_name:String} AND staled_at IS NULL
                )
                ORDER BY updated_at DESC, id DESC
                LIMIT {page_size:UInt32}
                OFFSET {offset:UInt32}
                FORMAT JSONEachRow");

                assert_eq!(parameters.get("dataset_name"), Some(&"test_dataset"));
                assert_eq!(parameters.get("page_size"), Some(&"10"));
                assert_eq!(parameters.get("offset"), Some(&"20"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"
                    {"id": "0199cff5-3130-7e90-815c-91219e1a2dae","type": "chat","function_name": "test_function","name": "test_name","episode_id": "test_episode_id","updated_at": "2021-01-01T00:00:00Z"}
                    {"id": "f11946d7-4986-43a7-b530-33e6dbba3817","type": "chat","function_name": "test_function","name": "test_name_2","updated_at": "2021-01-01T00:00:00Z"}
                    "#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let params = GetDatasetRowsParams {
            dataset_name: "test_dataset".to_string(),
            page_size: 10,
            offset: 20,
        };

        let rows = conn.get_dataset_rows(&params).await.unwrap();

        assert_eq!(rows.len(), 2, "Should return 2 rows");
        assert_eq!(rows[0].id, "0199cff5-3130-7e90-815c-91219e1a2dae");
        assert_eq!(rows[1].id, "f11946d7-4986-43a7-b530-33e6dbba3817");
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
                LIMIT {page_size:UInt32}
                OFFSET {offset:UInt32}
                FORMAT JSONEachRow");

                assert_eq!(parameters.get("function_name"), Some(&"test_function"));
                assert_eq!(parameters.get("page_size"), Some(&"10"));
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
            page_size: Some(10),
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
                assert_query_does_not_contain(query, "LIMIT {page_size:UInt32}");
                assert_query_does_not_contain(query, "OFFSET {offset:UInt32}");

                assert!(
                    !parameters.contains_key("function_name"),
                    "Should not provide function_name as a query parameter"
                );
                assert!(
                    !parameters.contains_key("page_size"),
                    "Should not provide page_size as a query parameter"
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
            page_size: None,
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
    async fn test_stale_datapoint_chat_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "INSERT INTO {table:Identifier}");
                assert_query_contains(query, "tool_params");
                assert_query_does_not_contain(query, "output_schema");
                assert_query_contains(query, "now64() as staled_at");
                assert_query_contains(query, "now64() as updated_at");
                assert_query_contains(query, "FROM {table:Identifier} FINAL");
                assert_query_contains(
                    query,
                    "WHERE dataset_name = {dataset_name:String} AND id = {datapoint_id:String}",
                );

                assert_eq!(parameters.get("table"), Some(&"ChatInferenceDatapoint"));
                assert_eq!(parameters.get("dataset_name"), Some(&"test_dataset"));
                assert_eq!(
                    parameters.get("datapoint_id"),
                    Some(&"123e4567-e89b-12d3-a456-426614174000")
                );

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 1,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let params = StaleDatapointParams {
            dataset_name: "test_dataset".to_string(),
            datapoint_id: Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap(),
            function_type: DatapointKind::Chat,
        };
        conn.stale_datapoint(&params).await.unwrap();
    }

    #[tokio::test]
    async fn test_stale_datapoint_json_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "INSERT INTO {table:Identifier}");
                assert_query_contains(query, "output_schema");
                assert_query_does_not_contain(query, "tool_params");
                assert_query_contains(query, "now64() as staled_at");
                assert_query_contains(query, "now64() as updated_at");
                assert_query_contains(query, "FROM {table:Identifier} FINAL");
                assert_query_contains(
                    query,
                    "WHERE dataset_name = {dataset_name:String} AND id = {datapoint_id:String}",
                );

                assert_eq!(parameters.get("table"), Some(&"JsonInferenceDatapoint"));
                assert_eq!(parameters.get("dataset_name"), Some(&"my_dataset"));
                assert_eq!(
                    parameters.get("datapoint_id"),
                    Some(&"223e4567-e89b-12d3-a456-426614174000")
                );

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 1,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let params = StaleDatapointParams {
            dataset_name: "my_dataset".to_string(),
            datapoint_id: Uuid::parse_str("223e4567-e89b-12d3-a456-426614174000").unwrap(),
            function_type: DatapointKind::Json,
        };
        conn.stale_datapoint(&params).await.unwrap();
    }

    #[tokio::test]
    async fn test_insert_chat_datapoint_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_write_non_batched_internal()
            .withf(|rows, table| {
                assert_eq!(
                    *table,
                    TableName::ChatInferenceDatapoint,
                    "Should write to ChatInferenceDatapoint"
                );

                let actual_row_as_json: serde_json::Value = serde_json::from_str(&rows[0]).unwrap();
                let expected_row_as_json = serde_json::json!({
                    "dataset_name": "test_dataset",
                    "function_name": "test_function",
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "name": "test_name",
                    "episode_id": "123e4567-e89b-12d3-a456-426614174000",
                    "input": {
                        "messages": [],
                    },
                    "output": [{"type": "text", "text": "response"}],
                    "tool_params": "",
                    "tags": {"test_tag": "test_value"},
                    "auxiliary": "",
                    "source_inference_id": null,
                    "is_custom": true,
                });
                assert_eq!(
                    actual_row_as_json, expected_row_as_json,
                    "Actual ChatInferenceDatapoint should match expected json value"
                );

                true
            })
            .returning(|_, _| Ok(()));
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
            conn.insert_datapoint(&DatapointInsert::Chat(datapoint))
                .await
                .is_ok(),
            "Should insert chat datapoint successfully"
        );
    }

    #[tokio::test]
    async fn test_insert_json_datapoint_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_write_non_batched_internal()
            .withf(|rows, table| {
                assert_eq!(
                    *table,
                    TableName::JsonInferenceDatapoint,
                    "Should write to JsonInferenceDatapoint"
                );

                let actual_row_as_json: serde_json::Value = serde_json::from_str(&rows[0]).unwrap();
                let expected_row_as_json = serde_json::json!({
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
                });

                assert_eq!(
                    actual_row_as_json, expected_row_as_json,
                    "Actual JsonInferenceDatapoint should match expected json value"
                );

                true
            })
            .returning(|_, _| Ok(()));
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
            conn.insert_datapoint(&DatapointInsert::Json(datapoint))
                .await
                .is_ok(),
            "Should insert json datapoint successfully"
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
    async fn test_get_adjacent_datapoint_ids_with_both_adjacent() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "
                WITH DatasetIds AS (
                    SELECT toUInt128(id) as id_uint FROM ChatInferenceDatapoint WHERE dataset_name = {dataset_name:String}
                    UNION ALL
                    SELECT toUInt128(id) as id_uint FROM JsonInferenceDatapoint WHERE dataset_name = {dataset_name:String}
                )
                SELECT
                    NULLIF(
                        (SELECT uint_to_uuid(min(id_uint)) FROM DatasetIds WHERE id_uint > toUInt128({datapoint_id:UUID})),
                        toUUID('00000000-0000-0000-0000-000000000000')
                    ) as next_id,
                    NULLIF(
                        (SELECT uint_to_uuid(max(id_uint)) FROM DatasetIds WHERE id_uint < toUInt128({datapoint_id:UUID})),
                        toUUID('00000000-0000-0000-0000-000000000000')
                    ) as previous_id
                FORMAT JSONEachRow");

                assert_eq!(parameters.get("dataset_name"), Some(&"test_dataset"));
                assert_eq!(parameters.get("datapoint_id"), Some(&"223e4567-e89b-12d3-a456-426614174000"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"next_id":"323e4567-e89b-12d3-a456-426614174000","previous_id":"123e4567-e89b-12d3-a456-426614174000"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let params = GetAdjacentDatapointIdsParams {
            dataset_name: "test_dataset".to_string(),
            datapoint_id: Uuid::parse_str("223e4567-e89b-12d3-a456-426614174000").unwrap(),
        };

        let result = conn.get_adjacent_datapoint_ids(&params).await.unwrap();

        assert_eq!(
            result.previous_id,
            Some(Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap())
        );
        assert_eq!(
            result.next_id,
            Some(Uuid::parse_str("323e4567-e89b-12d3-a456-426614174000").unwrap())
        );
    }

    #[tokio::test]
    async fn test_get_adjacent_datapoint_ids_with_only_next() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(
                        r#"{"next_id":"323e4567-e89b-12d3-a456-426614174000","previous_id":null}"#,
                    ),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let params = GetAdjacentDatapointIdsParams {
            dataset_name: "test_dataset".to_string(),
            datapoint_id: Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap(),
        };

        let result = conn.get_adjacent_datapoint_ids(&params).await.unwrap();

        assert_eq!(result.previous_id, None);
        assert_eq!(
            result.next_id,
            Some(Uuid::parse_str("323e4567-e89b-12d3-a456-426614174000").unwrap())
        );
    }

    #[tokio::test]
    async fn test_get_adjacent_datapoint_ids_with_only_previous() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(
                        r#"{"next_id":null,"previous_id":"123e4567-e89b-12d3-a456-426614174000"}"#,
                    ),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let params = GetAdjacentDatapointIdsParams {
            dataset_name: "test_dataset".to_string(),
            datapoint_id: Uuid::parse_str("323e4567-e89b-12d3-a456-426614174000").unwrap(),
        };

        let result = conn.get_adjacent_datapoint_ids(&params).await.unwrap();

        assert_eq!(
            result.previous_id,
            Some(Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap())
        );
        assert_eq!(result.next_id, None);
    }

    #[tokio::test]
    async fn test_get_adjacent_datapoint_ids_with_none() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"next_id":null,"previous_id":null}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let params = GetAdjacentDatapointIdsParams {
            dataset_name: "test_dataset".to_string(),
            datapoint_id: Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap(),
        };

        let result = conn.get_adjacent_datapoint_ids(&params).await.unwrap();

        assert_eq!(result.previous_id, None);
        assert_eq!(result.next_id, None);
    }

    #[tokio::test]
    async fn test_get_datapoint_chat_executes_successfully() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, parameters| {
                assert_query_contains(query, "WITH dataset as (");
                assert_query_contains(query, "'chat' as type");
                assert_query_contains(query, "FROM ChatInferenceDatapoint FINAL");
                assert_query_contains(query, "formatDateTime(updated_at, '%Y-%m-%dT%H:%i:%SZ') AS updated_at");
                assert_query_contains(query, "dataset_name = {dataset_name:String}");
                assert_query_contains(query, "staled_at IS NULL");
                assert_query_contains(query, "UNION ALL");
                assert_query_contains(query, "'json' as type");
                assert_query_contains(query, "FROM JsonInferenceDatapoint FINAL");
                assert_query_contains(query, "id = {datapoint_id:String}");
                assert_query_contains(query, "FORMAT JSONEachRow");

                assert_eq!(parameters.get("dataset_name"), Some(&"test_dataset"));
                assert_eq!(parameters.get("datapoint_id"), Some(&"123e4567-e89b-12d3-a456-426614174000"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"type":"chat","dataset_name":"test_dataset","function_name":"test_function","name":"test_name","id":"123e4567-e89b-12d3-a456-426614174000","episode_id":"223e4567-e89b-12d3-a456-426614174000","input":"{\"messages\":[]}","output":"[{\"type\":\"text\",\"text\":\"test output\"}]","tool_params":"{\"tools_available\":[],\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}","tags":{},"auxiliary":"{}","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z"}"#),
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

        if let Datapoint::Chat(datapoint) = result {
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
                assert_query_contains(query, "FROM ChatInferenceDatapoint FINAL");
                assert_query_contains(query, "UNION ALL");
                assert_query_contains(query, "FROM JsonInferenceDatapoint FINAL");
                assert_query_contains(query, "id = {datapoint_id:String}");
                assert_query_contains(query, "staled_at IS NULL");

                assert_eq!(parameters.get("dataset_name"), Some(&"json_dataset"));
                assert_eq!(parameters.get("datapoint_id"), Some(&"323e4567-e89b-12d3-a456-426614174000"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"type":"json","dataset_name":"json_dataset","function_name":"json_function","name":null,"id":"323e4567-e89b-12d3-a456-426614174000","episode_id":null,"input":"{\"messages\":[]}","output":"{\"parsed\":{\"key\":\"value\"}}","output_schema":"{\"type\":\"object\"}","tags":{},"auxiliary":"{}","source_inference_id":null,"is_deleted":false,"is_custom":true,"staled_at":null,"updated_at":"2023-01-01T00:00:00Z"}"#),
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

        if let Datapoint::Json(datapoint) = result {
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

                assert_eq!(parameters.get("dataset_name"), Some(&"json_dataset"));
                assert_eq!(
                    parameters.get("datapoint_id"),
                    Some(&"323e4567-e89b-12d3-a456-426614174000")
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
                assert_query_does_not_contain(query, "staled_at IS NULL");

                assert_eq!(parameters.get("dataset_name"), Some(&"json_dataset"));
                assert_eq!(
                    parameters.get("datapoint_id"),
                    Some(&"323e4567-e89b-12d3-a456-426614174000")
                );

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
}
