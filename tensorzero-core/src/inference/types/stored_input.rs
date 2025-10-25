use crate::config::Config;
use crate::endpoints::object_storage::get_object;
use crate::error::Error;
use crate::inference::types::file::Base64FileMetadata;
#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::stored_input_message_content_to_python;
use crate::inference::types::storage::StoragePath;
use crate::inference::types::Base64File;
use crate::inference::types::FileWithPath;
use crate::inference::types::ResolvedInput;
use crate::inference::types::ResolvedInputMessage;
use crate::inference::types::ResolvedInputMessageContent;
use crate::inference::types::StoredContentBlock;
use crate::inference::types::TemplateInput;
use crate::inference::types::{Role, Text, Thought, ToolCall, ToolResult};
use futures::future::try_join_all;
use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::serialize_to_dict;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// The input type that we directly store in ClickHouse.
/// This is almost identical to `ResolvedInput`, but without `File` data.
/// Only the object-storage path is actually stored in clickhouse
/// (which can be used to re-fetch the file and produce a `ResolvedInput`).
///
/// `StoredInputMessage` has a custom deserializer that addresses legacy data formats in the database.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct StoredInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(test, ts(optional))]
    pub system: Option<Value>,
    #[serde(default)]
    pub messages: Vec<StoredInputMessage>,
}

/// Abstracts over a `Config` (without an actual embedded gateway)
/// and an http `Client, so that we can call `reresolve` from `StoredInference`
/// and `evaluations`
pub trait StoragePathResolver {
    async fn resolve(&self, storage_path: StoragePath) -> Result<String, Error>;
}

impl StoragePathResolver for Config {
    async fn resolve(&self, storage_path: StoragePath) -> Result<String, Error> {
        Ok(get_object(self.object_store_info.as_ref(), storage_path)
            .await?
            .data)
    }
}

impl StoredInput {
    /// Converts a `StoredInput` to a `ResolvedInput` by fetching the file data
    /// for any nested `File`s.
    pub async fn reresolve(
        self,
        resolver: &impl StoragePathResolver,
    ) -> Result<ResolvedInput, Error> {
        Ok(ResolvedInput {
            system: self.system,
            messages: try_join_all(
                self.messages
                    .into_iter()
                    .map(|message| message.reresolve(resolver)),
            )
            .await?,
        })
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
/// `StoredInputMessage` has a custom deserializer that addresses legacy data formats in the database.
pub struct StoredInputMessage {
    pub role: Role,
    pub content: Vec<StoredInputMessageContent>,
}

impl StoredInputMessage {
    pub async fn reresolve(
        self,
        resolver: &impl StoragePathResolver,
    ) -> Result<ResolvedInputMessage, Error> {
        Ok(ResolvedInputMessage {
            role: self.role,
            content: try_join_all(
                self.content
                    .into_iter()
                    .map(|content| content.reresolve(resolver)),
            )
            .await?,
        })
    }
}

impl<'de> Deserialize<'de> for StoredInputMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Role,
            Content,
        }

        struct StoredInputMessageVisitor;

        impl<'de> Visitor<'de> for StoredInputMessageVisitor {
            type Value = StoredInputMessage;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct StoredInputMessage")
            }

            fn visit_map<V>(self, mut map: V) -> Result<StoredInputMessage, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut role: Option<Role> = None;
                let mut content: Option<Vec<Value>> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Role => {
                            if role.is_some() {
                                return Err(de::Error::duplicate_field("role"));
                            }
                            role = Some(map.next_value()?);
                        }
                        Field::Content => {
                            if content.is_some() {
                                return Err(de::Error::duplicate_field("content"));
                            }
                            content = Some(map.next_value()?);
                        }
                    }
                }

                let role = role.ok_or_else(|| de::Error::missing_field("role"))?;
                let content_values = content.ok_or_else(|| de::Error::missing_field("content"))?;

                // Transform legacy Text format to new format
                let transformed_content: Result<Vec<StoredInputMessageContent>, V::Error> =
                    content_values
                        .into_iter()
                        .map(|mut value| {
                            // Check if this is a legacy Text variant: {"type": "text", "value": ...}
                            if let Some(obj) = value.as_object_mut() {
                                if obj.get("type").and_then(|v| v.as_str()) == Some("text") {
                                    if let Some(val) = obj.remove("value") {
                                        // Convert based on value type
                                        match val {
                                            Value::String(text) => {
                                                // Convert to new format: {"type": "text", "text": "..."}
                                                obj.insert("text".to_string(), Value::String(text));
                                            }
                                            Value::Object(arguments) => {
                                                // Convert to Template format by constructing a new object
                                                let mut new_obj = serde_json::Map::new();
                                                new_obj.insert("type".to_string(), Value::String("template".to_string()));
                                                new_obj.insert("name".to_string(), Value::String(role.implicit_template_name().to_string()));
                                                new_obj.insert("arguments".to_string(), Value::Object(arguments));
                                                *obj = new_obj;
                                            }
                                            _ => {
                                                return Err(de::Error::custom(
                                                    r#"The `value` field in a `{"type": "text", "value": ... }` content block must be a string or object"#
                                                ));
                                            }
                                        }
                                    }
                                }
                            }

                            // Deserialize the transformed value
                            serde_json::from_value(value).map_err(de::Error::custom)
                        })
                        .collect();

                Ok(StoredInputMessage {
                    role,
                    content: transformed_content?,
                })
            }
        }

        const FIELDS: &[&str] = &["role", "content"];
        deserializer.deserialize_struct("StoredInputMessage", FIELDS, StoredInputMessageVisitor)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub enum StoredInputMessageContent {
    Text(Text),
    Template(TemplateInput),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    RawText {
        value: String,
    },
    Thought(Thought),
    #[serde(alias = "image")]
    File(Box<StoredFile>),
    Unknown {
        data: Value,
        model_provider_name: Option<String>,
    },
    // We may extend this in the future to include other types of content
}

impl StoredInputMessageContent {
    pub async fn reresolve(
        self,
        resolver: &impl StoragePathResolver,
    ) -> Result<ResolvedInputMessageContent, Error> {
        match self {
            StoredInputMessageContent::Text(text) => Ok(ResolvedInputMessageContent::Text(text)),
            StoredInputMessageContent::Template(template) => {
                Ok(ResolvedInputMessageContent::Template(template))
            }
            StoredInputMessageContent::ToolCall(tool_call) => {
                Ok(ResolvedInputMessageContent::ToolCall(tool_call))
            }
            StoredInputMessageContent::ToolResult(tool_result) => {
                Ok(ResolvedInputMessageContent::ToolResult(tool_result))
            }
            StoredInputMessageContent::RawText { value } => {
                Ok(ResolvedInputMessageContent::RawText { value })
            }
            StoredInputMessageContent::Thought(thought) => {
                Ok(ResolvedInputMessageContent::Thought(thought))
            }
            StoredInputMessageContent::File(file) => {
                let data = resolver.resolve(file.storage_path.clone()).await?;
                Ok(ResolvedInputMessageContent::File(Box::new(FileWithPath {
                    file: Base64File {
                        url: file.file.url.clone(),
                        mime_type: file.file.mime_type.clone(),
                        data,
                    },
                    storage_path: file.storage_path.clone(),
                })))
            }
            StoredInputMessageContent::Unknown {
                data,
                model_provider_name,
            } => Ok(ResolvedInputMessageContent::Unknown {
                data,
                model_provider_name,
            }),
        }
    }
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(get_all, str))]
pub struct StoredFile {
    #[serde(alias = "image")]
    pub file: Base64FileMetadata,
    pub storage_path: StoragePath,
}

impl std::fmt::Display for StoredInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl std::fmt::Display for StoredInputMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl std::fmt::Display for StoredFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl StoredInputMessage {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter]
    pub fn get_role(&self) -> String {
        self.role.to_string()
    }

    #[getter]
    pub fn get_content<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.content
            .iter()
            .map(|content| {
                stored_input_message_content_to_python(py, content.clone())
                    .map(|pyobj| pyobj.into_bound(py))
            })
            .collect()
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl StoredInput {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter]
    pub fn get_system<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(serialize_to_dict(py, self.system.clone())?.into_bound(py))
    }

    #[getter]
    pub fn get_messages(&self) -> Vec<StoredInputMessage> {
        self.messages.clone()
    }
}

/// The message type that we directly store in ClickHouse.
/// This is almost identical to `RequestMessage`, but without `File` data.
/// Only the object-storage path is actually stored in clickhouse
/// The `RequestMessage/StoredRequestMessage` pair is the model-level equivalent
/// of `ResolvedInput/StoredInput`
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StoredRequestMessage {
    pub role: Role,
    pub content: Vec<StoredContentBlock>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_deserialize_legacy_text_string() {
        // Legacy format with string value
        let json = json!({
            "role": "user",
            "content": [{"type": "text", "value": "Hello, world!"}]
        });

        let message: StoredInputMessage = serde_json::from_value(json).unwrap();
        assert_eq!(message.role, Role::User);
        assert_eq!(message.content.len(), 1);
        match &message.content[0] {
            StoredInputMessageContent::Text(text) => {
                assert_eq!(text.text, "Hello, world!");
            }
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_deserialize_legacy_text_object() {
        // Legacy format with object value (should convert to Template)
        let json = json!({
            "role": "user",
            "content": [{"type": "text", "value": {"foo": "bar", "baz": 123}}]
        });

        let message: StoredInputMessage = serde_json::from_value(json).unwrap();
        assert_eq!(message.role, Role::User);
        assert_eq!(message.content.len(), 1);
        match &message.content[0] {
            StoredInputMessageContent::Template(template) => {
                assert_eq!(template.name, "user");
                assert_eq!(template.arguments.get("foo").unwrap(), "bar");
                assert_eq!(template.arguments.get("baz").unwrap(), 123);
            }
            _ => panic!("Expected Template variant"),
        }
    }

    #[test]
    fn test_deserialize_new_text_format() {
        // New format with text field
        let json = json!({
            "role": "user",
            "content": [{"type": "text", "text": "Hello, world!"}]
        });

        let message: StoredInputMessage = serde_json::from_value(json).unwrap();
        assert_eq!(message.role, Role::User);
        assert_eq!(message.content.len(), 1);
        match &message.content[0] {
            StoredInputMessageContent::Text(text) => {
                assert_eq!(text.text, "Hello, world!");
            }
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_deserialize_legacy_text_invalid_value() {
        // Legacy format with invalid value type (number)
        let json = json!({
            "role": "user",
            "content": [{"type": "text", "value": 123}]
        });

        let result: Result<StoredInputMessage, _> = serde_json::from_value(json);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("must be a string or object"));
    }

    #[test]
    fn test_round_trip_serialization() {
        let message = StoredInputMessage {
            role: Role::User,
            content: vec![StoredInputMessageContent::Text(Text {
                text: "Hello, world!".to_string(),
            })],
        };

        let json = serde_json::to_value(&message).unwrap();
        let deserialized: StoredInputMessage = serde_json::from_value(json).unwrap();

        assert_eq!(message, deserialized);
    }
}
