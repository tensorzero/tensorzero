#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::Value;
use uuid::Uuid;

use crate::{
    inference::types::{ContentBlockChatOutput, JsonInferenceOutput, ResolvedInput},
    serde_util::{deserialize_defaulted_string_or_parsed_json, deserialize_string_or_parsed_json},
    tool::ToolCallConfigDatabaseInsert,
};

/// Represents an stored inference to be used for optimization.
/// These are retrieved from the database in this format.
/// NOTE / TODO: As an incremental step we are deserializing this enum from Python.
/// in the final version we should instead make this a native PyO3 class and
/// avoid deserialization entirely unless given a dict.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "pyo3", pyclass)]
pub enum StoredInference {
    Chat(StoredChatInference),
    Json(StoredJsonInference),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct StoredChatInference {
    pub function_name: String,
    pub variant_name: String,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: ResolvedInput,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub output: Vec<ContentBlockChatOutput>,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    #[serde(deserialize_with = "deserialize_defaulted_string_or_parsed_json")]
    pub tool_params: ToolCallConfigDatabaseInsert,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct StoredJsonInference {
    pub function_name: String,
    pub variant_name: String,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: ResolvedInput,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub output: JsonInferenceOutput,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub output_schema: Value,
}

impl StoredInference {
    pub fn input_mut(&mut self) -> &mut ResolvedInput {
        match self {
            StoredInference::Chat(example) => &mut example.input,
            StoredInference::Json(example) => &mut example.input,
        }
    }
    pub fn input(&self) -> &ResolvedInput {
        match self {
            StoredInference::Chat(example) => &example.input,
            StoredInference::Json(example) => &example.input,
        }
    }

    pub fn function_name(&self) -> &str {
        match self {
            StoredInference::Chat(example) => &example.function_name,
            StoredInference::Json(example) => &example.function_name,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_stored_inference_deserialization_chat() {
        // Test the ClickHouse version (doubly serialized)
        let json = r#"
            {
                "type": "chat",
                "function_name": "test_function",
                "variant_name": "test_variant",
                "input": "{\"system\": \"you are a helpful assistant\", \"messages\": []}",
                "output": "[{\"type\": \"text\", \"text\": \"Hello! How can I help you today?\"}]",
                "episode_id": "123e4567-e89b-12d3-a456-426614174000",
                "inference_id": "123e4567-e89b-12d3-a456-426614174000",
                "tool_params": ""
            }
        "#;
        let inference: StoredInference = serde_json::from_str(json).unwrap();
        let StoredInference::Chat(chat_inference) = inference else {
            panic!("Expected a chat inference");
        };
        assert_eq!(chat_inference.function_name, "test_function");
        assert_eq!(chat_inference.variant_name, "test_variant");
        assert_eq!(
            chat_inference.input,
            ResolvedInput {
                system: Some(json!("you are a helpful assistant")),
                messages: vec![],
            }
        );
        assert_eq!(
            chat_inference.output,
            vec!["Hello! How can I help you today?".to_string().into()]
        );

        // Test the Python version (singly serialized)
        let json = r#"
        {
            "type": "chat",
            "function_name": "test_function",
            "variant_name": "test_variant",
            "input": {"system": "you are a helpful assistant", "messages": []},
            "output": [{"type": "text", "text": "Hello! How can I help you today?"}],
            "episode_id": "123e4567-e89b-12d3-a456-426614174000",
            "inference_id": "123e4567-e89b-12d3-a456-426614174000",
            "tool_params": ""
        }
    "#;
        let inference: StoredInference = serde_json::from_str(json).unwrap();
        let StoredInference::Chat(chat_inference) = inference else {
            panic!("Expected a chat inference");
        };
        assert_eq!(chat_inference.function_name, "test_function");
        assert_eq!(chat_inference.variant_name, "test_variant");
        assert_eq!(
            chat_inference.input,
            ResolvedInput {
                system: Some(json!("you are a helpful assistant")),
                messages: vec![],
            }
        );
        assert_eq!(
            chat_inference.output,
            vec!["Hello! How can I help you today?".to_string().into()]
        );
    }

    #[test]
    fn test_stored_inference_deserialization_json() {
        // Test the ClickHouse version (doubly serialized)
        let json = r#"
            {
                "type": "json",
                "function_name": "test_function",
                "variant_name": "test_variant",
                "input": "{\"system\": \"you are a helpful assistant\", \"messages\": []}",
                "output": "{\"raw\":\"{\\\"answer\\\":\\\"Goodbye\\\"}\",\"parsed\":{\"answer\":\"Goodbye\"}}",
                "episode_id": "123e4567-e89b-12d3-a456-426614174000",
                "inference_id": "123e4567-e89b-12d3-a456-426614174000",
                "output_schema": "{\"type\": \"object\", \"properties\": {\"output\": {\"type\": \"string\"}}}"
            }
        "#;
        let inference: StoredInference = serde_json::from_str(json).unwrap();
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a json inference");
        };
        assert_eq!(json_inference.function_name, "test_function");
        assert_eq!(json_inference.variant_name, "test_variant");
        assert_eq!(
            json_inference.input,
            ResolvedInput {
                system: Some(json!("you are a helpful assistant")),
                messages: vec![],
            }
        );
        assert_eq!(
            json_inference.output,
            JsonInferenceOutput {
                raw: Some("{\"answer\":\"Goodbye\"}".to_string()),
                parsed: Some(json!({"answer":"Goodbye"})),
            }
        );
        assert_eq!(
            json_inference.episode_id,
            Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()
        );
        assert_eq!(
            json_inference.inference_id,
            Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()
        );
        assert_eq!(
            json_inference.output_schema,
            json!({"type": "object", "properties": {"output": {"type": "string"}}})
        );

        // Test the Python version (singly serialized)
        let json = r#"
         {
             "type": "json",
             "function_name": "test_function",
             "variant_name": "test_variant",
             "input": {"system": "you are a helpful assistant", "messages": []},
             "output": {"raw":"{\"answer\":\"Goodbye\"}","parsed":{"answer":"Goodbye"}},
             "episode_id": "123e4567-e89b-12d3-a456-426614174000",
             "inference_id": "123e4567-e89b-12d3-a456-426614174000",
             "output_schema": {"type": "object", "properties": {"output": {"type": "string"}}}
         }
     "#;
        let inference: StoredInference = serde_json::from_str(json).unwrap();
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a json inference");
        };
        assert_eq!(json_inference.function_name, "test_function");
        assert_eq!(json_inference.variant_name, "test_variant");
        assert_eq!(
            json_inference.input,
            ResolvedInput {
                system: Some(json!("you are a helpful assistant")),
                messages: vec![],
            }
        );
        assert_eq!(
            json_inference.output,
            JsonInferenceOutput {
                raw: Some("{\"answer\":\"Goodbye\"}".to_string()),
                parsed: Some(json!({"answer":"Goodbye"})),
            }
        );
        assert_eq!(
            json_inference.episode_id,
            Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()
        );
        assert_eq!(
            json_inference.inference_id,
            Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()
        );
        assert_eq!(
            json_inference.output_schema,
            json!({"type": "object", "properties": {"output": {"type": "string"}}})
        );
    }
}
