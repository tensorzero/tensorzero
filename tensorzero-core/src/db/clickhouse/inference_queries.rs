use std::collections::{BTreeMap, HashMap};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    db::TableBoundsWithCount,
    error::{Error, ErrorDetails},
    inference::types::FunctionType,
    serde_util::{deserialize_bool_from_integer, deserialize_option_u64, deserialize_u64},
};

use super::{ClickHouseConnectionInfo, ClickHouseResponse};

#[derive(Debug, Clone, Default, Serialize, Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct InferenceTableFilter {
    pub episode_id: Option<Uuid>,
    pub function_name: Option<String>,
    pub variant_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct QueryInferenceTableParams {
    pub page_size: u32,
    pub before: Option<Uuid>,
    pub after: Option<Uuid>,
    pub filter: Option<InferenceTableFilter>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct QueryInferenceTableBoundsParams {
    pub filter: Option<InferenceTableFilter>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct InferenceByIdRow {
    pub id: Uuid,
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub function_type: FunctionType,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct InferenceRow {
    pub id: Uuid,
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub input: String,
    pub output: String,
    pub tool_params: Option<String>,
    pub inference_params: String,
    #[serde(deserialize_with = "deserialize_u64")]
    pub processing_time_ms: u64,
    pub output_schema: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub tags: BTreeMap<String, String>,
    pub function_type: FunctionType,
    pub extra_body: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct ModelInferenceRow {
    pub id: Uuid,
    pub inference_id: Uuid,
    pub raw_request: String,
    pub raw_response: String,
    pub model_name: String,
    pub model_provider_name: String,
    #[serde(default, deserialize_with = "deserialize_option_u64")]
    pub input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_option_u64")]
    pub output_tokens: Option<u64>,
    #[serde(deserialize_with = "deserialize_u64")]
    pub response_time_ms: u64,
    #[serde(default, deserialize_with = "deserialize_option_u64")]
    pub ttft_ms: Option<u64>,
    pub timestamp: DateTime<Utc>,
    pub system: Option<String>,
    pub input_messages: String,
    pub output: String,
    #[serde(deserialize_with = "deserialize_bool_from_integer")]
    pub cached: bool,
    pub extra_body: Option<String>,
    pub tags: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct FunctionCountInfo {
    pub function_name: String,
    pub max_timestamp: DateTime<Utc>,
    #[serde(deserialize_with = "deserialize_u64")]
    pub count: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct TableBounds {
    pub first_id: Option<Uuid>,
    pub last_id: Option<Uuid>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct AdjacentIds {
    pub previous_id: Option<Uuid>,
    pub next_id: Option<Uuid>,
}

#[async_trait]
pub trait InferenceQueries {
    async fn query_inference_table(
        &self,
        params: &QueryInferenceTableParams,
    ) -> Result<Vec<InferenceByIdRow>, Error>;

    async fn query_inference_table_bounds(
        &self,
        params: &QueryInferenceTableBoundsParams,
    ) -> Result<TableBoundsWithCount, Error>;

    async fn count_inferences_for_function(
        &self,
        function_name: String,
        function_type: FunctionType,
    ) -> Result<u32, Error>;

    async fn count_inferences_for_variant(
        &self,
        function_name: String,
        function_type: FunctionType,
        variant_name: String,
    ) -> Result<u32, Error>;

    async fn count_inferences_for_episode(&self, episode_id: Uuid) -> Result<u32, Error>;

    async fn query_inference_by_id(&self, id: Uuid) -> Result<Option<InferenceRow>, Error>;

    async fn query_model_inferences_by_inference_id(
        &self,
        inference_id: Uuid,
    ) -> Result<Vec<ModelInferenceRow>, Error>;

    async fn count_inferences_by_function(&self) -> Result<Vec<FunctionCountInfo>, Error>;

    async fn get_adjacent_inference_ids(
        &self,
        current_inference_id: Uuid,
    ) -> Result<AdjacentIds, Error>;

    async fn get_adjacent_episode_ids(
        &self,
        current_episode_id: Uuid,
    ) -> Result<AdjacentIds, Error>;
}

fn parse_json_each_row<T: DeserializeOwned>(
    response: &ClickHouseResponse,
) -> Result<Vec<T>, Error> {
    parse_json_each_row_from_str(&response.response)
}

fn parse_json_each_row_from_str<T: DeserializeOwned>(s: &str) -> Result<Vec<T>, Error> {
    s.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<T>(line).map_err(|e| {
                Error::new(ErrorDetails::ClickHouseDeserialization {
                    message: e.to_string(),
                })
            })
        })
        .collect()
}

fn parse_single_row<T: DeserializeOwned>(
    response: &ClickHouseResponse,
) -> Result<Option<T>, Error> {
    let mut rows = parse_json_each_row::<T>(response)?;
    Ok(rows.pop())
}

fn build_where_clause(
    params: &Option<InferenceTableFilter>,
    where_clauses: &mut Vec<String>,
    param_values: &mut HashMap<String, String>,
) {
    if let Some(filter) = params {
        if let Some(episode_id) = filter.episode_id {
            where_clauses.push("episode_id = {episode_id:UUID}".to_string());
            param_values.insert("episode_id".to_string(), episode_id.to_string());
        }
        if let Some(function_name) = filter.function_name.as_ref() {
            where_clauses.push("function_name = {function_name:String}".to_string());
            param_values.insert("function_name".to_string(), function_name.clone());
        }
        if let Some(variant_name) = filter.variant_name.as_ref() {
            where_clauses.push("variant_name = {variant_name:String}".to_string());
            param_values.insert("variant_name".to_string(), variant_name.clone());
        }
    }
}

fn build_param_refs<'a>(values: &'a HashMap<String, String>) -> HashMap<&'a str, &'a str> {
    values
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect()
}

#[async_trait]
impl InferenceQueries for ClickHouseConnectionInfo {
    async fn query_inference_table(
        &self,
        params: &QueryInferenceTableParams,
    ) -> Result<Vec<InferenceByIdRow>, Error> {
        if params.before.is_some() && params.after.is_some() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Cannot specify both 'before' and 'after' parameters".to_string(),
            }));
        }

        let mut where_clauses = Vec::new();
        let mut param_values: HashMap<String, String> = HashMap::new();
        param_values.insert("page_size".to_string(), params.page_size.to_string());

        if let Some(before) = params.before {
            where_clauses.push("id_uint < toUInt128({before:UUID})".to_string());
            param_values.insert("before".to_string(), before.to_string());
        }

        if let Some(after) = params.after {
            where_clauses.push("id_uint > toUInt128({after:UUID})".to_string());
            param_values.insert("after".to_string(), after.to_string());
        }

        build_where_clause(&params.filter, &mut where_clauses, &mut param_values);

        let combined_where = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        let query = if params.after.is_some() {
            format!(
                r#"
      SELECT
        id,
        function_name,
        variant_name,
        episode_id,
        function_type,
        formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
      FROM
      (
        SELECT
          uint_to_uuid(id_uint) as id,
          id_uint,
          function_name,
          variant_name,
          episode_id,
          function_type,
          formatDateTime(UUIDv7ToDateTime(uint_to_uuid(id_uint)), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
        FROM InferenceById FINAL
        {combined_where}
        ORDER BY id_uint ASC
        LIMIT {{page_size:UInt32}}
      )
      ORDER BY id_uint DESC
      FORMAT JSONEachRow
    "#
            )
        } else {
            format!(
                r#"
      SELECT
        uint_to_uuid(id_uint) as id,
        function_name,
        variant_name,
        episode_id,
        function_type,
        formatDateTime(UUIDv7ToDateTime(uint_to_uuid(id_uint)), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
      FROM InferenceById FINAL
      {combined_where}
      ORDER BY id_uint DESC
      LIMIT {{page_size:UInt32}}
      FORMAT JSONEachRow
    "#
            )
        };

        let params_ref = build_param_refs(&param_values);
        let response = self.run_query_synchronous(query, &params_ref).await?;
        parse_json_each_row(&response)
    }

    async fn query_inference_table_bounds(
        &self,
        params: &QueryInferenceTableBoundsParams,
    ) -> Result<TableBoundsWithCount, Error> {
        let mut where_clauses = Vec::new();
        let mut param_values: HashMap<String, String> = HashMap::new();
        build_where_clause(&params.filter, &mut where_clauses, &mut param_values);

        let where_clause = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        let query = format!(
            r#"
  SELECT
    uint_to_uuid(MIN(id_uint)) AS first_id,
    uint_to_uuid(MAX(id_uint)) AS last_id,
    toUInt32(COUNT()) AS count
  FROM InferenceById FINAL
  {where_clause}
  LIMIT 1
  FORMAT JSONEachRow
  "#
        );

        let params_ref = build_param_refs(&param_values);
        let response = self.run_query_synchronous(query, &params_ref).await?;
        let mut rows = parse_json_each_row::<TableBoundsWithCount>(&response)?;
        if let Some(row) = rows.pop() {
            if row.count == 0 {
                Ok(TableBoundsWithCount {
                    first_id: None,
                    last_id: None,
                    count: 0,
                })
            } else {
                Ok(row)
            }
        } else {
            Ok(TableBoundsWithCount {
                first_id: None,
                last_id: None,
                count: 0,
            })
        }
    }

    async fn count_inferences_for_function(
        &self,
        function_name: String,
        function_type: FunctionType,
    ) -> Result<u32, Error> {
        let table_name = function_type.inference_table_name();
        let query = format!(
            "SELECT toUInt32(COUNT()) AS count FROM {table_name} WHERE function_name = {{function_name:String}} FORMAT JSONEachRow"
        );
        let mut param_values = HashMap::new();
        param_values.insert("function_name".to_string(), function_name);
        let params_ref = build_param_refs(&param_values);
        let response = self.run_query_synchronous(query, &params_ref).await?;
        let mut rows = parse_json_each_row::<CountRow>(&response)?;
        let count = rows.pop().map(|row| row.as_u32()).unwrap_or_default();
        Ok(count)
    }

    async fn count_inferences_for_variant(
        &self,
        function_name: String,
        function_type: FunctionType,
        variant_name: String,
    ) -> Result<u32, Error> {
        let table_name = function_type.inference_table_name();
        let query = format!(
            "SELECT toUInt32(COUNT()) AS count FROM {table_name} WHERE function_name = {{function_name:String}} AND variant_name = {{variant_name:String}} FORMAT JSONEachRow"
        );
        let mut param_values = HashMap::new();
        param_values.insert("function_name".to_string(), function_name);
        param_values.insert("variant_name".to_string(), variant_name);
        let params_ref = build_param_refs(&param_values);
        let response = self.run_query_synchronous(query, &params_ref).await?;
        let mut rows = parse_json_each_row::<CountRow>(&response)?;
        let count = rows.pop().map(|row| row.as_u32()).unwrap_or_default();
        Ok(count)
    }

    async fn count_inferences_for_episode(&self, episode_id: Uuid) -> Result<u32, Error> {
        let query = "SELECT toUInt32(COUNT()) AS count FROM InferenceByEpisodeId FINAL WHERE episode_id_uint = toUInt128({episode_id:UUID}) FORMAT JSONEachRow".to_string();
        let mut param_values = HashMap::new();
        param_values.insert("episode_id".to_string(), episode_id.to_string());
        let params_ref = build_param_refs(&param_values);
        let response = self.run_query_synchronous(query, &params_ref).await?;
        let mut rows = parse_json_each_row::<CountRow>(&response)?;
        let count = rows.pop().map(|row| row.as_u32()).unwrap_or_default();
        Ok(count)
    }

    async fn query_inference_by_id(&self, id: Uuid) -> Result<Option<InferenceRow>, Error> {
        let query = r#"
    WITH inference AS (
        SELECT
            id_uint,
            function_name,
            variant_name,
            episode_id,
            function_type
        FROM InferenceById
        WHERE id_uint = toUInt128({id:UUID})
        LIMIT 1
    )
    SELECT
        c.id,
        c.function_name,
        c.variant_name,
        c.episode_id,
        c.input,
        c.output,
        c.tool_params,
        c.inference_params,
        c.processing_time_ms,
        NULL AS output_schema,
        formatDateTime(c.timestamp, '%Y-%m-%dT%H:%i:%SZ') AS timestamp,
        c.tags,
        'chat' AS function_type,
        c.extra_body
    FROM ChatInference c
    WHERE
        'chat' = (SELECT function_type FROM inference)
        AND c.function_name IN (SELECT function_name FROM inference)
        AND c.variant_name IN (SELECT variant_name FROM inference)
        AND c.episode_id IN (SELECT episode_id FROM inference)
        AND c.id = {id:UUID}

    UNION ALL

    SELECT
        j.id,
        j.function_name,
        j.variant_name,
        j.episode_id,
        j.input,
        j.output,
        NULL AS tool_params,
        j.inference_params,
        j.processing_time_ms,
        j.output_schema,
        formatDateTime(j.timestamp, '%Y-%m-%dT%H:%i:%SZ') AS timestamp,
        j.tags,
        'json' AS function_type,
        j.extra_body
    FROM JsonInference j
    WHERE
        'json' = (SELECT function_type FROM inference)
        AND j.function_name IN (SELECT function_name FROM inference)
        AND j.variant_name IN (SELECT variant_name FROM inference)
        AND j.episode_id IN (SELECT episode_id FROM inference)
        AND j.id = {id:UUID}
    FORMAT JSONEachRow
  "#
        .to_string();
        let mut param_values = HashMap::new();
        param_values.insert("id".to_string(), id.to_string());
        let params_ref = build_param_refs(&param_values);
        let response = self.run_query_synchronous(query, &params_ref).await?;
        parse_single_row(&response)
    }

    async fn query_model_inferences_by_inference_id(
        &self,
        inference_id: Uuid,
    ) -> Result<Vec<ModelInferenceRow>, Error> {
        let query = "SELECT *, formatDateTime(timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp FROM ModelInference WHERE inference_id = {id:UUID} FORMAT JSONEachRow".to_string();
        let mut param_values = HashMap::new();
        param_values.insert("id".to_string(), inference_id.to_string());
        let params_ref = build_param_refs(&param_values);
        let response = self.run_query_synchronous(query, &params_ref).await?;
        parse_json_each_row(&response)
    }

    async fn count_inferences_by_function(&self) -> Result<Vec<FunctionCountInfo>, Error> {
        let query = r#"SELECT
        function_name,
        formatDateTime(max(timestamp), '%Y-%m-%dT%H:%i:%SZ') AS max_timestamp,
        toUInt32(count()) AS count
    FROM (
        SELECT function_name, timestamp
        FROM ChatInference
        UNION ALL
        SELECT function_name, timestamp
        FROM JsonInference
    )
    GROUP BY function_name
    ORDER BY max_timestamp DESC
    FORMAT JSONEachRow"#
            .to_string();
        let response = self.run_query_synchronous_no_params(query).await?;
        parse_json_each_row(&response)
    }

    async fn get_adjacent_inference_ids(
        &self,
        current_inference_id: Uuid,
    ) -> Result<AdjacentIds, Error> {
        let query = r#"
    SELECT
      NULLIF(
        (SELECT uint_to_uuid(max(id_uint)) FROM InferenceById WHERE id_uint < toUInt128({current_inference_id:UUID})),
        toUUID('00000000-0000-0000-0000-000000000000')
      ) as previous_id,
      NULLIF(
        (SELECT uint_to_uuid(min(id_uint)) FROM InferenceById WHERE id_uint > toUInt128({current_inference_id:UUID})),
        toUUID('00000000-0000-0000-0000-000000000000')
      ) as next_id
    FORMAT JSONEachRow
  "#
        .to_string();
        let mut param_values = HashMap::new();
        param_values.insert(
            "current_inference_id".to_string(),
            current_inference_id.to_string(),
        );
        let params_ref = build_param_refs(&param_values);
        let response = self.run_query_synchronous(query, &params_ref).await?;
        let mut rows = parse_json_each_row::<AdjacentIds>(&response)?;
        Ok(rows.pop().unwrap_or_default())
    }

    async fn get_adjacent_episode_ids(
        &self,
        current_episode_id: Uuid,
    ) -> Result<AdjacentIds, Error> {
        let query = r#"
    SELECT
      NULLIF(
        (SELECT DISTINCT uint_to_uuid(max(episode_id_uint)) FROM InferenceByEpisodeId WHERE episode_id_uint < toUInt128({current_episode_id:UUID})),
        toUUID('00000000-0000-0000-0000-000000000000')
      ) as previous_id,
      NULLIF(
        (SELECT DISTINCT uint_to_uuid(min(episode_id_uint)) FROM InferenceByEpisodeId WHERE episode_id_uint > toUInt128({current_episode_id:UUID})),
        toUUID('00000000-0000-0000-0000-000000000000')
      ) as next_id
    FORMAT JSONEachRow
  "#
        .to_string();
        let mut param_values = HashMap::new();
        param_values.insert(
            "current_episode_id".to_string(),
            current_episode_id.to_string(),
        );
        let params_ref = build_param_refs(&param_values);
        let response = self.run_query_synchronous(query, &params_ref).await?;
        let mut rows = parse_json_each_row::<AdjacentIds>(&response)?;
        Ok(rows.pop().unwrap_or_default())
    }
}

#[derive(Debug, Deserialize)]
struct CountRow {
    #[serde(deserialize_with = "deserialize_u64")]
    count: u64,
}

impl CountRow {
    fn as_u32(&self) -> u32 {
        self.count as u32
    }
}
