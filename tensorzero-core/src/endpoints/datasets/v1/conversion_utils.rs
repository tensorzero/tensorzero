use futures::future::join_all;
use uuid::Uuid;

use crate::config::Config;
use crate::db::datasets::{ChatInferenceDatapointInsert, JsonInferenceDatapointInsert};
use crate::endpoints::datasets::v1::types::{
    CreateChatDatapointRequest, CreateJsonDatapointRequest, JsonDatapointOutputUpdate,
};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::{FetchContext, JsonInferenceOutput};
use crate::jsonschema_util::{DynamicJSONSchema, JsonSchemaRef};
use crate::tool::ToolCallConfigDatabaseInsert;

impl CreateChatDatapointRequest {
    /// Validates and converts this request into a ChatInferenceDatapointInsert struct.
    pub async fn into_database_insert(
        self,
        config: &Config,
        fetch_context: &FetchContext<'_>,
        dataset_name: &str,
    ) -> Result<ChatInferenceDatapointInsert, Error> {
        // Validate function exists and is a chat function
        let function_config = config.get_function(&self.function_name)?;
        let FunctionConfig::Chat(_) = &**function_config else {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!(
                    "Function '{}' is not configured as a chat function",
                    self.function_name
                ),
            }));
        };

        // Validate and convert input
        function_config.validate_input(&self.input)?;
        let stored_input = self
            .input
            .into_lazy_resolved_input(fetch_context)?
            .into_stored_input(fetch_context.object_store_info)
            .await?;

        // Validate and re-parse the raw fields in InferenceResponseToolCall blocks against the tool call config.
        let tool_config =
            function_config.prepare_tool_config(self.dynamic_tool_params, &config.tools)?;

        let validated_output = if let Some(output) = self.output {
            let validation_futures = output
                .into_iter()
                .map(|output| output.into_validated(tool_config.as_ref()));
            let validated_output = join_all(validation_futures).await;
            Some(validated_output)
        } else {
            None
        };

        let insert = ChatInferenceDatapointInsert {
            dataset_name: dataset_name.to_string(),
            function_name: self.function_name,
            name: self.name,
            id: Uuid::now_v7(),
            episode_id: self.episode_id,
            input: stored_input,
            output: validated_output,
            tool_params: tool_config.map(ToolCallConfigDatabaseInsert::from),
            tags: self.tags,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };

        Ok(insert)
    }
}

impl CreateJsonDatapointRequest {
    /// Validates and converts this request into a JsonInferenceDatapointInsert struct.
    pub async fn into_database_insert(
        self,
        config: &Config,
        fetch_context: &FetchContext<'_>,
        dataset_name: &str,
    ) -> Result<JsonInferenceDatapointInsert, Error> {
        // Validate function exists and is a JSON function
        let function_config = config.get_function(&self.function_name)?;
        let FunctionConfig::Json(json_function_config) = &**function_config else {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!(
                    "Function '{}' is not configured as a JSON function",
                    self.function_name
                ),
            }));
        };

        // Validate and convert input
        function_config.validate_input(&self.input)?;
        let stored_input = self
            .input
            .into_lazy_resolved_input(fetch_context)?
            .into_stored_input(fetch_context.object_store_info)
            .await?;

        // Determine the output schema (use provided or default to function's schema)
        // parsed_schema is declared here to make it live long enough until the end of the function.
        let parsed_schema: DynamicJSONSchema;
        let (output_schema, schema_ref) = if let Some(user_schema) = self.output_schema {
            // Validate the user-provided output_schema
            let schema_str = serde_json::to_string(&user_schema).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to serialize output_schema: {e}"),
                })
            })?;
            parsed_schema = DynamicJSONSchema::parse_from_str(&schema_str).map_err(|e| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Invalid output_schema: {e}"),
                })
            })?;
            // Ensure the schema is valid by forcing compilation
            parsed_schema.ensure_valid().await?;
            (user_schema, JsonSchemaRef::Dynamic(&parsed_schema))
        } else {
            (
                json_function_config.output_schema.value.clone(),
                JsonSchemaRef::Static(&json_function_config.output_schema),
            )
        };

        // Convert output.
        // We will validate the output against schema, because we allow users to create datapoints that
        // do not conform to the output schema (or does not even parse as valid JSON) by design.
        let output = match self.output {
            Some(output) => Some(output.into_json_inference_output(schema_ref).await),
            None => None,
        };

        let insert = JsonInferenceDatapointInsert {
            dataset_name: dataset_name.to_string(),
            function_name: self.function_name,
            name: self.name,
            id: Uuid::now_v7(),
            episode_id: self.episode_id,
            input: stored_input,
            output,
            output_schema,
            tags: self.tags,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };

        Ok(insert)
    }
}

impl JsonDatapointOutputUpdate {
    /// Converts this `JsonDatapointOutputUpdate` into a `JsonInferenceOutput`.
    ///
    /// This function parses and validates the `raw` output against the `output_schema`, and only
    /// populates the `parsed` field if the output is valid.
    pub async fn into_json_inference_output(
        self,
        output_schema: JsonSchemaRef<'_>,
    ) -> JsonInferenceOutput {
        let mut output = JsonInferenceOutput {
            raw: self.raw,
            parsed: None,
        };

        let Some(raw) = output.raw.as_ref() else {
            return output;
        };

        let parse_result = serde_json::from_str(raw.as_str());

        let Ok(parsed_unvalidated_value) = parse_result else {
            return output;
        };
        let Ok(()) = output_schema.validate(&parsed_unvalidated_value).await else {
            return output;
        };

        output.parsed = Some(parsed_unvalidated_value);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::{Config, ConfigFileGlob};
    use crate::http::TensorzeroHttpClient;
    use crate::inference::types::{
        Arguments, ContentBlockChatOutput, Input, InputMessage, InputMessageContent,
        JsonInferenceOutput, System, Template, Text,
    };
    use crate::inference::types::{Role, StoredInputMessage, StoredInputMessageContent};
    use crate::jsonschema_util::DynamicJSONSchema;
    use crate::tool::{DynamicToolParams, InferenceResponseToolCall};
    use serde_json::json;
    use std::collections::HashMap;
    use std::path::Path;
    use uuid::Uuid;

    async fn get_e2e_config() -> Config {
        Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(Path::new("tests/e2e/config/tensorzero.*.toml"))
                .unwrap(),
            false,
        )
        .await
        .unwrap()
        .config
    }

    fn create_test_input() -> Input {
        let mut args_map = serde_json::Map::new();
        args_map.insert("topic".to_string(), json!("nature"));

        Input {
            system: Some(System::Text("You are a helpful AI assistant.".to_string())),
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments: Arguments(args_map),
                })],
            }],
        }
    }

    fn create_test_output() -> Vec<ContentBlockChatOutput> {
        vec![ContentBlockChatOutput::Text(Text {
            text: "Test output".to_string(),
        })]
    }

    #[tokio::test]
    async fn test_chat_datapoint_conversion_all_fields() {
        let config = get_e2e_config().await;
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let fetch_context = FetchContext {
            client: &http_client,
            object_store_info: &None,
        };

        let episode_id = Uuid::now_v7();

        let mut tags = HashMap::new();
        tags.insert("environment".to_string(), "test".to_string());
        tags.insert("version".to_string(), "1.0".to_string());

        let request = CreateChatDatapointRequest {
            function_name: "write_haiku".to_string(),
            input: create_test_input(),
            output: Some(create_test_output()),
            dynamic_tool_params: DynamicToolParams::default(),
            episode_id: Some(episode_id),
            name: Some("Test Datapoint".to_string()),
            tags: Some(tags.clone()),
        };

        let insert = request
            .into_database_insert(&config, &fetch_context, "test_dataset")
            .await
            .unwrap();

        assert_eq!(insert.episode_id, Some(episode_id));
        assert_eq!(insert.tags, Some(tags));
        assert_eq!(insert.name, Some("Test Datapoint".to_string()));
        assert_eq!(insert.function_name, "write_haiku");
        assert_eq!(
            insert.input.system.unwrap(),
            System::Text("You are a helpful AI assistant.".to_string())
        );
        assert_eq!(
            insert.input.messages,
            vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments: Arguments(serde_json::Map::from_iter([(
                        "topic".to_string(),
                        json!("nature"),
                    )])),
                })],
            }]
        );
        assert_eq!(insert.output, Some(create_test_output()));
        assert!(insert.is_custom);
        assert!(insert.source_inference_id.is_none());
    }

    #[tokio::test]
    async fn test_chat_datapoint_invalid_function() {
        let config = get_e2e_config().await;
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let fetch_context = FetchContext {
            client: &http_client,
            object_store_info: &None,
        };

        let request = CreateChatDatapointRequest {
            function_name: "nonexistent_function".to_string(),
            input: create_test_input(),
            output: None,
            dynamic_tool_params: DynamicToolParams::default(),
            episode_id: None,
            name: None,
            tags: None,
        };

        let result = request
            .into_database_insert(&config, &fetch_context, "test_dataset")
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_chat_datapoint_wrong_function_type() {
        let config = get_e2e_config().await;
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let fetch_context = FetchContext {
            client: &http_client,
            object_store_info: &None,
        };

        // Try to use a JSON function for a chat request
        let request = CreateChatDatapointRequest {
            function_name: "extract_entities".to_string(), // This is a JSON function
            input: create_test_input(),
            output: None,
            dynamic_tool_params: DynamicToolParams::default(),
            episode_id: None,
            name: None,
            tags: None,
        };

        let result = request
            .into_database_insert(&config, &fetch_context, "test_dataset")
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("not configured as a chat function"));
    }

    #[tokio::test]
    async fn test_chat_datapoint_stores_only_raw_tool_output_for_unknown_tools() {
        let config = get_e2e_config().await;
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let fetch_context = FetchContext {
            client: &http_client,
            object_store_info: &None,
        };

        let episode_id = Uuid::now_v7();

        let mut tags = HashMap::new();
        tags.insert("environment".to_string(), "test".to_string());
        tags.insert("version".to_string(), "1.0".to_string());

        // This tool call is not present in either function config or dynamic tool params, so we store the raw output
        // and drop any arguments and names before going to database.
        let outputs = vec![ContentBlockChatOutput::ToolCall(
            InferenceResponseToolCall {
                arguments: Some(json!({"foo": "bar"})),
                id: "test_id".to_string(),
                name: Some("unknown_tool_name".to_string()),
                raw_arguments: r#"{"foo": "bar"}"#.to_string(),
                raw_name: "unknown_tool_name".to_string(),
            },
        )];

        let request = CreateChatDatapointRequest {
            function_name: "write_haiku".to_string(),
            input: create_test_input(),
            output: Some(outputs),
            dynamic_tool_params: DynamicToolParams::default(),
            episode_id: Some(episode_id),
            name: Some("Test Datapoint".to_string()),
            tags: Some(tags.clone()),
        };

        let insert = request
            .into_database_insert(&config, &fetch_context, "test_dataset")
            .await
            .unwrap();

        // We should drop the name and arguments before going to database.
        assert_eq!(
            insert.output,
            Some(vec![ContentBlockChatOutput::ToolCall(
                InferenceResponseToolCall {
                    arguments: None,
                    id: "test_id".to_string(),
                    name: None,
                    raw_arguments: r#"{"foo": "bar"}"#.to_string(),
                    raw_name: "unknown_tool_name".to_string(),
                }
            )])
        );
    }

    #[tokio::test]
    async fn test_json_datapoint_invalid_output_against_schema() {
        let config = get_e2e_config().await;
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let fetch_context = FetchContext {
            client: &http_client,
            object_store_info: &None,
        };

        // Create a simple input without templates for extract_entities
        let simple_input = Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Extract entities from this text".to_string(),
                })],
            }],
        };

        let request = CreateJsonDatapointRequest {
            function_name: "extract_entities".to_string(),
            input: simple_input,
            output: Some(JsonDatapointOutputUpdate {
                raw: Some(
                    serde_json::to_string(&json!({
                        "invalid_field": "this doesn't match the schema"
                    }))
                    .unwrap(),
                ),
            }),
            output_schema: None,
            episode_id: None,
            name: None,
            tags: None,
        };

        let result = request
            .into_database_insert(&config, &fetch_context, "test_dataset")
            .await;

        // Invalid output should be accepted, but parsed should be None due to schema validation failure.
        let insert = result.unwrap();
        assert_eq!(
            insert.output,
            Some(JsonInferenceOutput {
                raw: Some("{\"invalid_field\":\"this doesn't match the schema\"}".to_string()),
                parsed: None,
            })
        );
    }

    #[tokio::test]
    async fn test_json_datapoint_nonconformant_json_raw_value() {
        let config = get_e2e_config().await;
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let fetch_context = FetchContext {
            client: &http_client,
            object_store_info: &None,
        };

        // Create a simple input without templates for extract_entities
        let simple_input = Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Extract entities from this text".to_string(),
                })],
            }],
        };

        let request = CreateJsonDatapointRequest {
            function_name: "extract_entities".to_string(),
            input: simple_input,
            output: Some(JsonDatapointOutputUpdate {
                raw: Some("intentionally \" nonconformant json".to_string()),
            }),
            output_schema: None,
            episode_id: None,
            name: None,
            tags: None,
        };

        let result = request
            .into_database_insert(&config, &fetch_context, "test_dataset")
            .await;

        // Invalid JSON output should be accepted, with parsed set to None.
        let insert = result.unwrap();
        assert_eq!(
            insert.output,
            Some(JsonInferenceOutput {
                raw: Some("intentionally \" nonconformant json".to_string()),
                parsed: None,
            })
        );
    }

    #[tokio::test]
    async fn test_json_datapoint_invalid_function() {
        let config = get_e2e_config().await;
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let fetch_context = FetchContext {
            client: &http_client,
            object_store_info: &None,
        };

        let request = CreateJsonDatapointRequest {
            function_name: "nonexistent_function".to_string(),
            input: create_test_input(),
            output: None,
            output_schema: None,
            episode_id: None,
            name: None,
            tags: None,
        };

        let result = request
            .into_database_insert(&config, &fetch_context, "test_dataset")
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_json_datapoint_into_database_insert_basic() {
        let config = get_e2e_config().await;
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let fetch_context = FetchContext {
            client: &http_client,
            object_store_info: &None,
        };

        // Create a simple input without templates for extract_entities
        let simple_input = Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Extract entities from this text".to_string(),
                })],
            }],
        };

        let request = CreateJsonDatapointRequest {
            function_name: "extract_entities".to_string(),
            input: simple_input,
            output: Some(JsonDatapointOutputUpdate {
                raw: Some(
                    serde_json::to_string(&json!({
                        "person": ["Alice", "Bob"],
                        "organization": ["ACME Corp"],
                        "location": ["New York"],
                        "miscellaneous": []
                    }))
                    .unwrap(),
                ),
            }),
            output_schema: None,
            episode_id: None,
            name: Some("Test JSON Datapoint".to_string()),
            tags: None,
        };

        let result = request
            .into_database_insert(&config, &fetch_context, "test_dataset")
            .await;

        assert!(result.is_ok());
        let insert = result.unwrap();
        assert_eq!(insert.dataset_name, "test_dataset");
        assert_eq!(insert.function_name, "extract_entities");
        assert_eq!(insert.name, Some("Test JSON Datapoint".to_string()));
        assert!(insert.output.is_some());
        assert!(insert.is_custom);
    }

    #[tokio::test]
    async fn test_json_datapoint_wrong_function_type() {
        let config = get_e2e_config().await;
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let fetch_context = FetchContext {
            client: &http_client,
            object_store_info: &None,
        };

        // Try to use a chat function for a JSON request
        let request = CreateJsonDatapointRequest {
            function_name: "write_haiku".to_string(), // This is a chat function
            input: create_test_input(),
            output: None,
            output_schema: None,
            episode_id: None,
            name: None,
            tags: None,
        };

        let result = request
            .into_database_insert(&config, &fetch_context, "test_dataset")
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("not configured as a JSON function"));
    }

    #[tokio::test]
    async fn test_json_datapoint_output_update_into_json_inference_output_valid() {
        let update = JsonDatapointOutputUpdate {
            raw: Some(r#"{"key": "value"}"#.to_string()),
        };
        let schema_value = json!({"type": "object", "properties": {"key": {"type": "string"}}, });
        let schema = DynamicJSONSchema::new(schema_value);
        let schema_ref = JsonSchemaRef::Dynamic(&schema);
        let output = update.into_json_inference_output(schema_ref).await;

        assert_eq!(
            output.raw,
            Some(r#"{"key": "value"}"#.to_string()),
            "Raw field should be the same as the input"
        );
        assert_eq!(
            output.parsed,
            Some(json!({"key": "value"})),
            "Parsed field should be the same as the input because it conforms to the schema"
        );
    }

    #[tokio::test]
    async fn test_json_datapoint_output_update_into_json_inference_output_nonconformant() {
        let update = JsonDatapointOutputUpdate {
            raw: Some(r#"{"key": "nonconformant value"}"#.to_string()),
        };
        let schema_value = json!({"type": "object", "properties": {"key": {"type": "number"}}, });
        let schema = DynamicJSONSchema::new(schema_value);
        let schema_ref = JsonSchemaRef::Dynamic(&schema);
        let output = update.into_json_inference_output(schema_ref).await;
        assert_eq!(output.parsed, None);

        assert_eq!(
            output.raw,
            Some(r#"{"key": "nonconformant value"}"#.to_string()),
            "Raw field should be the same as the input"
        );
        assert_eq!(
            output.parsed, None,
            "Parsed field should be None because it does not conform to the schema"
        );
    }

    #[tokio::test]
    async fn test_json_datapoint_output_update_into_json_inference_output_invalid_json() {
        let update = JsonDatapointOutputUpdate {
            raw: Some("intentionally invalid \" json".to_string()),
        };

        let schema_value = json!({"type": "object", "properties": {"value": {"type": "string"}}, });
        let schema = DynamicJSONSchema::new(schema_value);
        let schema_ref = JsonSchemaRef::Dynamic(&schema);
        let output = update.into_json_inference_output(schema_ref).await;
        assert_eq!(output.parsed, None);

        assert_eq!(
            output.raw,
            Some("intentionally invalid \" json".to_string()),
            "Raw field should be the same as the input"
        );
        assert_eq!(
            output.parsed, None,
            "Parsed field should be None because it is invalid JSON"
        );
    }
}
