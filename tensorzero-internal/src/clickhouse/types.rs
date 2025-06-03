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
#[derive(Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredInference {
    Chat(StoredChatInference),
    Json(StoredJsonInference),
}

// TODO: test deserialization of these two types from strings and from parsed json
#[derive(Debug, Deserialize, PartialEq)]
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

#[derive(Debug, Deserialize, PartialEq)]
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
