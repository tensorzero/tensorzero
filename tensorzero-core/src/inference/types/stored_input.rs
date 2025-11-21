use crate::client::{File, InputMessage, InputMessageContent};
use crate::config::Config;
use crate::endpoints::object_storage::get_object;
use crate::error::Error;
use crate::inference::types::file::{Base64FileMetadata, ObjectStorageFile, ObjectStoragePointer};
#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::stored_input_message_content_to_python;
use crate::inference::types::storage::StoragePath;
use crate::inference::types::Input;
use crate::inference::types::ResolvedInput;
use crate::inference::types::ResolvedInputMessage;
use crate::inference::types::ResolvedInputMessageContent;
use crate::inference::types::StoredContentBlock;
use crate::inference::types::System;
use crate::inference::types::Template;
use crate::inference::types::{RawText, Role, Text, Thought, ToolCall, ToolResult, Unknown};
use crate::tool::ToolCallWrapper;
use futures::future::try_join_all;
use schemars::JsonSchema;
use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::ops::Deref;
use tensorzero_derive::export_schema;

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
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default, ts_rs::TS, JsonSchema)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[ts(export)]
#[export_schema]
pub struct StoredInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub system: Option<System>,
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

    /// Converts a `StoredInput` to an `Input` without fetching file data.
    /// Files are converted to `File::ObjectStoragePointer` variant which contains
    /// only metadata (source_url, mime_type, storage_path) without the actual file data.
    ///
    /// TODO(shuyangli): Add optional parameter to fetch files from object storage.
    pub fn into_input(self) -> Input {
        Input {
            system: self.system,
            messages: self
                .messages
                .into_iter()
                .map(StoredInputMessage::into_input_message)
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, ts_rs::TS, JsonSchema)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[ts(export)]
#[export_schema]
/// `StoredInputMessage` has a custom deserializer that addresses legacy data formats in the database (see below).
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

    /// Converts a `StoredInputMessage` to an `InputMessage`, possibly fetching files (later).
    pub fn into_input_message(self) -> InputMessage {
        InputMessage {
            role: self.role,
            content: self
                .content
                .into_iter()
                .map(StoredInputMessageContent::into_input_message_content)
                .collect(),
        }
    }
}

/// `StoredInputMessage` has a custom deserializer that addresses legacy data formats in the database:
///
/// - {"type": "text", "value": "..."} -> {"type": "text", "text": "..."}
/// - {"type": "text", "value": { ... }} -> {"type": "template", "name": "role", "arguments": { ... }}
///
/// The correct type {"type": "text", "text": "..."} goes through without modification.
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, ts_rs::TS, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
#[ts(export)]
#[export_schema]
pub enum StoredInputMessageContent {
    #[schemars(title = "StoredInputMessageContentText")]
    Text(Text),
    #[schemars(title = "StoredInputMessageContentTemplate")]
    Template(Template),
    #[schemars(title = "StoredInputMessageContentToolCall")]
    ToolCall(ToolCall),
    #[schemars(title = "StoredInputMessageContentToolResult")]
    ToolResult(ToolResult),
    #[schemars(title = "StoredInputMessageContentRawText")]
    RawText(RawText),
    #[schemars(title = "StoredInputMessageContentThought")]
    Thought(Thought),
    #[serde(alias = "image")]
    #[schemars(title = "StoredInputMessageContentFile", with = "ObjectStoragePointer")]
    File(Box<StoredFile>),
    #[schemars(title = "StoredInputMessageContentUnknown")]
    Unknown(Unknown),
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
            StoredInputMessageContent::RawText(raw_text) => {
                Ok(ResolvedInputMessageContent::RawText(raw_text))
            }
            StoredInputMessageContent::Thought(thought) => {
                Ok(ResolvedInputMessageContent::Thought(thought))
            }
            StoredInputMessageContent::File(file) => {
                let data = resolver.resolve(file.storage_path.clone()).await?;
                Ok(ResolvedInputMessageContent::File(Box::new(
                    ObjectStorageFile {
                        file: ObjectStoragePointer {
                            source_url: file.source_url.clone(),
                            mime_type: file.mime_type.clone(),
                            storage_path: file.storage_path.clone(),
                            detail: file.detail.clone(),
                            filename: file.filename.clone(),
                        },
                        data,
                    },
                )))
            }
            StoredInputMessageContent::Unknown(unknown) => {
                Ok(ResolvedInputMessageContent::Unknown(unknown))
            }
        }
    }

    /// Converts a `StoredInputMessageContent` to the client type `InputMessageContent`, possibly fetching files (later).
    pub fn into_input_message_content(self) -> InputMessageContent {
        match self {
            StoredInputMessageContent::Text(text) => InputMessageContent::Text(text),
            StoredInputMessageContent::Template(template) => {
                InputMessageContent::Template(template)
            }
            StoredInputMessageContent::ToolCall(tool_call) => {
                InputMessageContent::ToolCall(ToolCallWrapper::ToolCall(tool_call))
            }
            StoredInputMessageContent::ToolResult(tool_result) => {
                InputMessageContent::ToolResult(tool_result)
            }
            StoredInputMessageContent::RawText(raw_text) => InputMessageContent::RawText(raw_text),
            StoredInputMessageContent::Thought(thought) => InputMessageContent::Thought(thought),
            StoredInputMessageContent::File(stored_file) => {
                // Convert StoredFile (ObjectStoragePointer) to File::ObjectStoragePointer
                // This preserves only the metadata without fetching actual file data
                InputMessageContent::File(File::ObjectStoragePointer(stored_file.0))
            }
            StoredInputMessageContent::Unknown(unknown) => InputMessageContent::Unknown(unknown),
        }
    }
}

/// A newtype wrapper around `ObjectStoragePointer` that handles legacy deserialization formats.
/// See the deserializer implementation below for details on the legacy formats it supports.
#[derive(Clone, Debug, Serialize, PartialEq, ts_rs::TS)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[repr(transparent)]
#[serde(transparent)]
pub struct StoredFile(pub ObjectStoragePointer);

impl Deref for StoredFile {
    type Target = ObjectStoragePointer;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<ObjectStoragePointer> for StoredFile {
    fn from(file: ObjectStoragePointer) -> Self {
        Self(file)
    }
}

impl From<StoredFile> for ObjectStoragePointer {
    fn from(file: StoredFile) -> Self {
        file.0
    }
}

/// Implement a custom deserializer for `StoredFile` to handle legacy formats
///
/// The custom deserializer handles:
///
/// - Legacy nested format: `{ file: { source_url, mime_type }, storage_path }`
/// - Legacy `image` alias: `{ image: { source_url, mime_type }, storage_path }`
/// - Deprecated `url` field (via `ObjectStorageFile`): `{ url: ..., mime_type, storage_path }`
/// - New flattened format: `{ source_url, mime_type, storage_path }` (delegates to `ObjectStorageFile`)
impl<'de> Deserialize<'de> for StoredFile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct LegacyStoredFile {
            #[serde(alias = "image")]
            file: Base64FileMetadata,
            storage_path: StoragePath,
        }

        let value = serde_json::Value::deserialize(deserializer)?;

        // Check if this is the legacy nested format (has `file` or `image` field)
        if value.get("file").is_some() || value.get("image").is_some() {
            let legacy: LegacyStoredFile =
                serde_json::from_value(value).map_err(de::Error::custom)?;

            return Ok(StoredFile(ObjectStoragePointer {
                source_url: legacy.file.source_url,
                mime_type: legacy.file.mime_type,
                storage_path: legacy.storage_path,
                detail: None,
                filename: None,
            }));
        }

        // For the new flattened format, delegate to `ObjectStorageFile`'s deserializer
        // which already handles the `url` vs `source_url` alias deprecation
        let file: ObjectStoragePointer =
            serde_json::from_value(value).map_err(de::Error::custom)?;
        Ok(StoredFile(file))
    }
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
impl StoredFile {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter(source_url)]
    pub fn source_url_string(&self) -> Option<String> {
        self.0
            .source_url
            .as_ref()
            .map(std::string::ToString::to_string)
    }

    #[getter(mime_type)]
    pub fn mime_type_string(&self) -> String {
        self.0.mime_type.to_string()
    }

    #[getter]
    pub fn get_storage_path(&self) -> StoragePath {
        self.0.storage_path.clone()
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
                assert_eq!(template.arguments.0.get("foo").unwrap(), "bar");
                assert_eq!(template.arguments.0.get("baz").unwrap(), 123);
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

    #[test]
    fn test_deserialize_stored_file_legacy_nested_format() {
        use crate::inference::types::storage::StorageKind;

        // Legacy format with nested "file" field
        let json = json!({
            "file": {
                "source_url": "https://example.com/image.png",
                "mime_type": "image/png"
            },
            "storage_path": {
                "kind": {
                    "type": "disabled"
                },
                "path": "test/image.png"
            }
        });

        let stored_file: StoredFile = serde_json::from_value(json).unwrap();
        assert_eq!(
            stored_file.source_url.as_ref().unwrap().as_str(),
            "https://example.com/image.png"
        );
        assert_eq!(stored_file.mime_type, mime::IMAGE_PNG);
        assert!(matches!(
            stored_file.storage_path.kind,
            StorageKind::Disabled
        ));
    }

    #[test]
    fn test_deserialize_stored_file_legacy_nested_format_with_image_alias() {
        use crate::inference::types::storage::StorageKind;

        // Legacy format with nested "image" field (alias)
        let json = json!({
            "image": {
                "source_url": "https://example.com/photo.jpg",
                "mime_type": "image/jpeg"
            },
            "storage_path": {
                "kind": {
                    "type": "disabled"
                },
                "path": "test/photo.jpg"
            }
        });

        let stored_file: StoredFile = serde_json::from_value(json).unwrap();
        assert_eq!(
            stored_file.source_url.as_ref().unwrap().as_str(),
            "https://example.com/photo.jpg"
        );
        assert_eq!(stored_file.mime_type, mime::IMAGE_JPEG);
        assert!(matches!(
            stored_file.storage_path.kind,
            StorageKind::Disabled
        ));
    }

    #[test]
    fn test_deserialize_stored_file_new_flattened_format() {
        use crate::inference::types::storage::StorageKind;

        // New flattened format
        let json = json!({
            "source_url": "https://example.com/image.png",
            "mime_type": "image/png",
            "storage_path": {
                "kind": {
                    "type": "disabled"
                },
                "path": "test/image.png"
            }
        });

        let stored_file: StoredFile = serde_json::from_value(json).unwrap();
        assert_eq!(
            stored_file.source_url.as_ref().unwrap().as_str(),
            "https://example.com/image.png"
        );
        assert_eq!(stored_file.mime_type, mime::IMAGE_PNG);
        assert!(matches!(
            stored_file.storage_path.kind,
            StorageKind::Disabled
        ));
    }

    #[test]
    fn test_deserialize_stored_file_with_url_alias() {
        use crate::inference::types::storage::StorageKind;

        // New format with deprecated "url" field
        let json = json!({
            "url": "https://example.com/image.png",
            "mime_type": "image/png",
            "storage_path": {
                "kind": {
                    "type": "disabled"
                },
                "path": "test/image.png"
            }
        });

        let stored_file: StoredFile = serde_json::from_value(json).unwrap();
        assert_eq!(
            stored_file.source_url.as_ref().unwrap().as_str(),
            "https://example.com/image.png"
        );
        assert_eq!(stored_file.mime_type, mime::IMAGE_PNG);
        assert!(matches!(
            stored_file.storage_path.kind,
            StorageKind::Disabled
        ));
    }

    #[test]
    fn test_serialize_stored_file_always_flattened() {
        use crate::inference::types::storage::{StorageKind, StoragePath};

        let stored_file = StoredFile(ObjectStoragePointer {
            source_url: Some("https://example.com/image.png".parse().unwrap()),
            mime_type: mime::IMAGE_PNG,
            storage_path: StoragePath {
                kind: StorageKind::Disabled,
                path: object_store::path::Path::parse("test/image.png").unwrap(),
            },
            detail: None,
            filename: None,
        });

        let json = serde_json::to_value(&stored_file).unwrap();

        // Serialization should always produce flattened format
        assert!(json.get("source_url").is_some());
        assert!(json.get("mime_type").is_some());
        assert!(json.get("storage_path").is_some());
        assert!(json.get("file").is_none());
        assert!(json.get("image").is_none());
    }

    #[test]
    fn test_round_trip_stored_file_serialization() {
        use crate::inference::types::storage::{StorageKind, StoragePath};

        let original = StoredFile(ObjectStoragePointer {
            source_url: Some("https://example.com/test.png".parse().unwrap()),
            mime_type: mime::IMAGE_PNG,
            storage_path: StoragePath {
                kind: StorageKind::Disabled,
                path: object_store::path::Path::parse("test/path.png").unwrap(),
            },
            detail: None,
            filename: None,
        });

        let json = serde_json::to_value(&original).unwrap();
        let deserialized: StoredFile = serde_json::from_value(json).unwrap();

        assert_eq!(original, deserialized);
    }
}
