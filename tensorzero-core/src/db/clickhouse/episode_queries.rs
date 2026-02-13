use async_trait::async_trait;
use serde::Deserialize;
use uuid::Uuid;

use crate::config::Config;
use crate::db::clickhouse::query_builder::parameters::add_parameter;
use crate::db::clickhouse::query_builder::{ClickhouseType, JoinRegistry, QueryParameter};
use crate::db::{EpisodeByIdRow, EpisodeQueries, TableBoundsWithCount};
use crate::endpoints::stored_inferences::v1::types::InferenceFilter;
use crate::error::{Error, ErrorDetails};
use crate::serde_util::deserialize_u64;

use super::ClickHouseConnectionInfo;

#[async_trait]
impl EpisodeQueries for ClickHouseConnectionInfo {
    async fn query_episode_table(
        &self,
        config: &Config,
        limit: u32,
        before: Option<Uuid>,
        after: Option<Uuid>,
        function_name: Option<String>,
        filters: Option<InferenceFilter>,
    ) -> Result<Vec<EpisodeByIdRow>, Error> {
        let (query, params_map) = build_episode_query(
            config,
            limit,
            before,
            after,
            function_name.as_deref(),
            filters.as_ref(),
        )?;
        let params_str_map: std::collections::HashMap<&str, &str> = params_map
            .iter()
            .map(|p| (p.name.as_str(), p.value.as_str()))
            .collect();
        let response = self.run_query_synchronous(query, &params_str_map).await?;
        // Deserialize the results into EpisodeByIdRow
        response
            .response
            .trim()
            .lines()
            .map(|row| {
                serde_json::from_str(row).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: e.to_string(),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()
    }

    async fn query_episode_table_bounds(&self) -> Result<TableBoundsWithCount, Error> {
        let query = r"
            SELECT
                uint_to_uuid(min(episode_id_uint)) as first_id,
                uint_to_uuid(max(episode_id_uint)) as last_id,
                count() as count
            FROM EpisodeById
            FORMAT JSONEachRow"
            .to_string();
        let response = self.run_query_synchronous_no_params(query).await?;
        let response = response.response.trim();
        serde_json::from_str(response).map_err(|e| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: e.to_string(),
            })
        })
    }
}

/// Builds the ClickHouse query and parameters for listing episodes.
///
/// When `function_name` or `filters` are provided, a CTE filters episode IDs
/// by querying the inference tables for matching inferences.
fn build_episode_query(
    config: &Config,
    limit: u32,
    before: Option<Uuid>,
    after: Option<Uuid>,
    function_name: Option<&str>,
    filters: Option<&InferenceFilter>,
) -> Result<(String, Vec<QueryParameter>), Error> {
    let mut params_map: Vec<QueryParameter> = Vec::new();
    let mut param_idx_counter: usize = 0;

    let (where_clause, should_reverse_results) = match (before, after) {
        (Some(_before), Some(_after)) => {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Cannot specify both before and after in query_episode_table".to_string(),
            }));
        }
        (Some(before), None) => {
            let param = add_parameter(
                before,
                ClickhouseType::String,
                &mut params_map,
                &mut param_idx_counter,
            );
            (
                format!("episode_id_uint < toUInt128(toUUID({param}))"),
                false,
            )
        }
        (None, Some(after)) => {
            let param = add_parameter(
                after,
                ClickhouseType::String,
                &mut params_map,
                &mut param_idx_counter,
            );
            (
                format!("episode_id_uint > toUInt128(toUUID({param}))"),
                true,
            )
        }
        (None, None) => ("1=1".to_string(), false),
    };

    let has_filters = function_name.is_some() || filters.is_some();

    // Build the filtered_episodes CTE if we have function_name or filters
    let filtered_episodes_cte = if has_filters {
        build_filtered_episodes_cte(
            config,
            function_name,
            filters,
            &mut params_map,
            &mut param_idx_counter,
        )?
    } else {
        String::new()
    };

    let filtered_episodes_condition = if has_filters {
        "AND episode_id_uint IN (SELECT episode_id_uint FROM filtered_episodes)"
    } else {
        ""
    };

    // This is a Bad Hack to account for ClickHouse's Bad Query Planning
    // Clickhouse will not optimize queries with either GROUP BY or FINAL
    // when reading the primary key with a LIMIT
    // https://clickhouse.com/docs/sql-reference/statements/select/order-by#optimization-of-data-reading
    // So, we select the last limit_overestimate episodes in descending order
    // Then we group by episode_id_uint and count the number of inferences
    // Finally, we order by episode_id_uint correctly and limit the result to limit
    let limit_overestimate = 5 * limit;

    let with_prefix = if has_filters {
        format!("WITH {filtered_episodes_cte},")
    } else {
        "WITH".to_string()
    };

    let query = if should_reverse_results {
        format!(
            r"
            {with_prefix} potentially_duplicated_episode_ids AS (
                SELECT episode_id_uint
                FROM EpisodeById
                WHERE {where_clause}
                {filtered_episodes_condition}
                ORDER BY episode_id_uint ASC
                LIMIT {limit_overestimate}
            )
            SELECT
                episode_id,
                count,
                start_time,
                end_time,
                last_inference_id
            FROM (
                SELECT
                    uint_to_uuid(episode_id_uint) as episode_id,
                    count() as count,
                    formatDateTime(UUIDv7ToDateTime(uint_to_uuid(min(id_uint))), '%Y-%m-%dT%H:%i:%SZ') as start_time,
                    formatDateTime(UUIDv7ToDateTime(uint_to_uuid(max(id_uint))), '%Y-%m-%dT%H:%i:%SZ') as end_time,
                    uint_to_uuid(max(id_uint)) as last_inference_id,
                    episode_id_uint
                FROM InferenceByEpisodeId
                WHERE episode_id_uint IN (SELECT episode_id_uint FROM potentially_duplicated_episode_ids)
                GROUP BY episode_id_uint
                ORDER BY episode_id_uint ASC
                LIMIT {limit}
            )
            ORDER BY episode_id_uint DESC
            FORMAT JSONEachRow
            "
        )
    } else {
        format!(
            r"
            {with_prefix} potentially_duplicated_episode_ids AS (
                SELECT episode_id_uint
                FROM EpisodeById
                WHERE {where_clause}
                {filtered_episodes_condition}
                ORDER BY episode_id_uint DESC
                LIMIT {limit_overestimate}
            )
            SELECT
                uint_to_uuid(episode_id_uint) as episode_id,
                count() as count,
                formatDateTime(UUIDv7ToDateTime(uint_to_uuid(min(id_uint))), '%Y-%m-%dT%H:%i:%SZ') as start_time,
                formatDateTime(UUIDv7ToDateTime(uint_to_uuid(max(id_uint))), '%Y-%m-%dT%H:%i:%SZ') as end_time,
                uint_to_uuid(max(id_uint)) as last_inference_id
            FROM InferenceByEpisodeId
            WHERE episode_id_uint IN (SELECT episode_id_uint FROM potentially_duplicated_episode_ids)
            GROUP BY episode_id_uint
            ORDER BY episode_id_uint DESC
            LIMIT {limit}
            FORMAT JSONEachRow
            "
        )
    };

    Ok((query, params_map))
}

/// Builds the filtered_episodes CTE that finds episode IDs matching the given criteria.
///
/// For function_name only, we can use the InferenceByEpisodeId table directly.
/// For complex filters, we query from ChatInference/JsonInference which have all columns.
fn build_filtered_episodes_cte(
    config: &Config,
    function_name: Option<&str>,
    filters: Option<&InferenceFilter>,
    params_map: &mut Vec<QueryParameter>,
    param_idx_counter: &mut usize,
) -> Result<String, Error> {
    if filters.is_some() {
        // With filters, we need to query from ChatInference/JsonInference
        // because InferenceByEpisodeId doesn't have tags, timestamp, etc.
        build_filtered_episodes_from_inference_tables(
            config,
            function_name,
            filters,
            params_map,
            param_idx_counter,
        )
    } else if let Some(fn_name) = function_name {
        // function_name only - use InferenceByEpisodeId for efficiency
        let fn_param = add_parameter(
            fn_name,
            ClickhouseType::String,
            params_map,
            param_idx_counter,
        );
        Ok(format!(
            r"filtered_episodes AS (
                SELECT DISTINCT episode_id_uint
                FROM InferenceByEpisodeId
                WHERE function_name = {fn_param}
            )"
        ))
    } else {
        // This branch should never be reached since the caller checks has_filters
        Err(Error::new(ErrorDetails::InvalidRequest {
            message: "build_filtered_episodes_cte called without function_name or filters"
                .to_string(),
        }))
    }
}

/// Builds the filtered_episodes CTE by querying ChatInference and JsonInference tables.
/// This is needed when filters reference columns not available in InferenceByEpisodeId.
fn build_filtered_episodes_from_inference_tables(
    config: &Config,
    function_name: Option<&str>,
    filters: Option<&InferenceFilter>,
    params_map: &mut Vec<QueryParameter>,
    param_idx_counter: &mut usize,
) -> Result<String, Error> {
    // Build filter conditions for ChatInference
    let mut chat_joins = JoinRegistry::new();
    let mut chat_select_clauses = Vec::new();
    let mut chat_where_clauses: Vec<String> = Vec::new();

    if let Some(fn_name) = function_name {
        let fn_param = add_parameter(
            fn_name,
            ClickhouseType::String,
            params_map,
            param_idx_counter,
        );
        chat_where_clauses.push(format!("i.function_name = {fn_param}"));
    }

    if let Some(filter) = filters {
        let filter_sql = filter.to_clickhouse_sql(
            config,
            params_map,
            &mut chat_select_clauses,
            &mut chat_joins,
            param_idx_counter,
        )?;
        chat_where_clauses.push(filter_sql);
    }

    let chat_where = if chat_where_clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", chat_where_clauses.join(" AND "))
    };

    let chat_joins_sql = chat_joins.get_clauses().join("\n");

    // Build filter conditions for JsonInference
    let mut json_joins = JoinRegistry::new();
    let mut json_select_clauses = Vec::new();
    let mut json_where_clauses: Vec<String> = Vec::new();

    if let Some(fn_name) = function_name {
        let fn_param = add_parameter(
            fn_name,
            ClickhouseType::String,
            params_map,
            param_idx_counter,
        );
        json_where_clauses.push(format!("i.function_name = {fn_param}"));
    }

    if let Some(filter) = filters {
        let filter_sql = filter.to_clickhouse_sql(
            config,
            params_map,
            &mut json_select_clauses,
            &mut json_joins,
            param_idx_counter,
        )?;
        json_where_clauses.push(filter_sql);
    }

    let json_where = if json_where_clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", json_where_clauses.join(" AND "))
    };

    let json_joins_sql = json_joins.get_clauses().join("\n");

    Ok(format!(
        r"filtered_episodes AS (
            SELECT DISTINCT toUInt128(episode_id) AS episode_id_uint
            FROM ChatInference AS i
            {chat_joins_sql}
            {chat_where}
            UNION ALL
            SELECT DISTINCT toUInt128(episode_id) AS episode_id_uint
            FROM JsonInference AS i
            {json_joins_sql}
            {json_where}
        )"
    ))
}

// Helper functions for parsing responses
pub(crate) fn build_pagination_clause(
    before: Option<Uuid>,
    after: Option<Uuid>,
    _id_column: &str,
) -> (String, Vec<(&'static str, String)>) {
    match (before, after) {
        (Some(before), None) => (
            "AND toUInt128(id) < toUInt128(toUUID({before:UUID}))".to_string(),
            vec![("before", before.to_string())],
        ),
        (None, Some(after)) => (
            "AND toUInt128(id) > toUInt128(toUUID({after:UUID}))".to_string(),
            vec![("after", after.to_string())],
        ),
        _ => (String::new(), vec![]),
    }
}

pub(crate) fn parse_json_rows<T: serde::de::DeserializeOwned>(
    response: &str,
) -> Result<Vec<T>, Error> {
    response
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|row| {
            serde_json::from_str(row).map_err(|e| {
                Error::new(ErrorDetails::ClickHouseDeserialization {
                    message: format!("Failed to deserialize row: {e}"),
                })
            })
        })
        .collect()
}

pub(crate) fn parse_count(response: &str) -> Result<u64, Error> {
    #[derive(Deserialize)]
    struct CountResult {
        #[serde(deserialize_with = "deserialize_u64")]
        count: u64,
    }

    let line = response.trim().lines().next().ok_or_else(|| {
        Error::new(ErrorDetails::ClickHouseDeserialization {
            message: "No count result returned from database".to_string(),
        })
    })?;

    let result: CountResult = serde_json::from_str(line).map_err(|e| {
        Error::new(ErrorDetails::ClickHouseDeserialization {
            message: format!("Failed to deserialize count: {e}"),
        })
    })?;

    Ok(result.count)
}
