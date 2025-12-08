//! ClickHouse queries for inference statistics.

use std::collections::HashMap;

use super::ClickHouseConnectionInfo;
use super::select_queries::parse_count;
use crate::error::Error;
use crate::function::FunctionConfigType;

/// Parameters for counting inferences for a function.
#[derive(Debug)]
pub struct CountInferencesParams<'a> {
    pub function_name: &'a str,
    pub function_type: FunctionConfigType,
    pub variant_name: Option<&'a str>,
}

impl ClickHouseConnectionInfo {
    /// Counts the number of inferences for a function, optionally filtered by variant.
    pub async fn count_inferences_for_function(
        &self,
        params: CountInferencesParams<'_>,
    ) -> Result<u64, Error> {
        let (query, params) = build_count_inferences_query(&params);
        let response = self.run_query_synchronous(query, &params).await?;
        parse_count(&response.response)
    }
}

/// Builds the SQL query for counting inferences.
fn build_count_inferences_query<'a>(
    params: &'a CountInferencesParams<'a>,
) -> (String, HashMap<&'a str, &'a str>) {
    let mut query_params = HashMap::new();
    query_params.insert("function_name", params.function_name);

    let table_name = match params.function_type {
        FunctionConfigType::Chat => "ChatInference",
        FunctionConfigType::Json => "JsonInference",
    };

    let query = match params.variant_name {
        Some(variant_name) => {
            query_params.insert("variant_name", variant_name);
            format!(
                "SELECT COUNT() AS count
         FROM {table_name}
         WHERE function_name = {{function_name:String}}
           AND variant_name = {{variant_name:String}}
         FORMAT JSONEachRow"
            )
        }
        None => {
            format!(
                "SELECT COUNT() AS count
         FROM {table_name}
         WHERE function_name = {{function_name:String}}
         FORMAT JSONEachRow"
            )
        }
    };

    (query, query_params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::clickhouse::query_builder::test_util::{
        assert_query_contains, assert_query_does_not_contain,
    };

    #[test]
    fn test_build_count_inferences_query_chat_no_variant() {
        let params = CountInferencesParams {
            function_name: "write_haiku",
            function_type: FunctionConfigType::Chat,
            variant_name: None,
        };
        let (query, query_params) = build_count_inferences_query(&params);
        assert_query_contains(&query, "FROM ChatInference");
        assert_query_contains(&query, "function_name = {function_name:String}");
        assert_query_does_not_contain(&query, "variant_name");
        assert_eq!(query_params.len(), 1);
        assert_eq!(query_params.get("function_name"), Some(&"write_haiku"));
    }

    #[test]
    fn test_build_count_inferences_query_json_no_variant() {
        let params = CountInferencesParams {
            function_name: "extract_entities",
            function_type: FunctionConfigType::Json,
            variant_name: None,
        };
        let (query, query_params) = build_count_inferences_query(&params);
        assert_query_contains(&query, "FROM JsonInference");
        assert_query_contains(&query, "function_name = {function_name:String}");
        assert_query_does_not_contain(&query, "variant_name");
        assert_eq!(query_params.len(), 1);
        assert_eq!(query_params.get("function_name"), Some(&"extract_entities"));
    }

    #[test]
    fn test_build_count_inferences_query_chat_with_variant() {
        let params = CountInferencesParams {
            function_name: "write_haiku",
            function_type: FunctionConfigType::Chat,
            variant_name: Some("initial_prompt_gpt4o_mini"),
        };
        let (query, query_params) = build_count_inferences_query(&params);
        assert_query_contains(&query, "FROM ChatInference");
        assert_query_contains(&query, "function_name = {function_name:String}");
        assert_query_contains(&query, "variant_name = {variant_name:String}");
        assert_eq!(query_params.len(), 2);
        assert_eq!(query_params.get("function_name"), Some(&"write_haiku"));
        assert_eq!(
            query_params.get("variant_name"),
            Some(&"initial_prompt_gpt4o_mini")
        );
    }

    #[test]
    fn test_build_count_inferences_query_json_with_variant() {
        let params = CountInferencesParams {
            function_name: "extract_entities",
            function_type: FunctionConfigType::Json,
            variant_name: Some("gpt4o_initial_prompt"),
        };
        let (query, query_params) = build_count_inferences_query(&params);
        assert_query_contains(&query, "FROM JsonInference");
        assert_query_contains(&query, "function_name = {function_name:String}");
        assert_query_contains(&query, "variant_name = {variant_name:String}");
        assert_eq!(query_params.len(), 2);
        assert_eq!(query_params.get("function_name"), Some(&"extract_entities"));
        assert_eq!(
            query_params.get("variant_name"),
            Some(&"gpt4o_initial_prompt")
        );
    }
}
