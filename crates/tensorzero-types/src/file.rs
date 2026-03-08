//! File types for inference inputs.
//!
//! This module defines file types for different stages of the file lifecycle:
//! from client requests through storage.

use crate::error::TypeError;
use crate::storage::StoragePath;
use mime::MediaType;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero_derive::export_schema;
use url::Url;

/// Detail level for input images (affects fidelity and token cost)
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub enum Detail {
    Low,
    High,
    Auto,
}

/// A file already encoded as base64
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
#[cfg_attr(feature = "pyo3", pyclass)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub struct Base64File {
    // The original url we used to download the file
    #[serde(alias = "url")] // DEPRECATED
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[schemars(with = "Option<String>")]
    pub source_url: Option<Url>,
    #[cfg_attr(feature = "ts-bindings", ts(type = "string"))]
    #[schemars(with = "String")]
    pub mime_type: MediaType,
    // This field contains *unprefixed* base64-encoded data.
    // It's private and validated by the constructor.
    data: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub detail: Option<Detail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub filename: Option<String>,
}

impl Base64File {
    /// Create a new Base64File with validation.
    ///
    /// Returns an error if the data contains a `data:` prefix.
    /// If `mime_type` is not provided, we will try to detect it from the file data.
    pub fn new(
        source_url: Option<Url>,
        mime_type: Option<MediaType>,
        data: String,
        detail: Option<Detail>,
        filename: Option<String>,
    ) -> Result<Self, TypeError> {
        if data.starts_with("data:") {
            return Err(TypeError::InvalidDataPrefix(
                "The `data` field must not contain `data:` prefix. Data should be pure base64-encoded content only.".to_string(),
            ));
        }

        let mime_type = if let Some(mime_type) = mime_type {
            mime_type
        } else {
            // Decode base64 and infer mime type from the data
            let decoded = aws_smithy_types::base64::decode(&data).map_err(|e| {
                TypeError::InvalidBase64(format!("Failed to decode base64 data: {e}"))
            })?;

            let inferred = infer::get(&decoded);
            if let Some(inferred_type) = inferred {
                inferred_type
                    .mime_type()
                    .parse::<MediaType>()
                    .map_err(|e| {
                        TypeError::InvalidMimeType(format!("Inferred mime type is not valid: {e}"))
                    })?
            } else {
                return Err(TypeError::InvalidMimeType(
                    "No mime type provided and unable to infer from data".to_string(),
                ));
            }
        };

        Ok(Self {
            source_url,
            mime_type,
            data,
            detail,
            filename,
        })
    }

    pub fn data(&self) -> &str {
        &self.data
    }

    /// Create a new Base64File from pre-validated parts.
    ///
    /// This is used when you already have valid base64 data (e.g., you just encoded it)
    /// and a known mime type. Unlike `new`, this does not validate the data format
    /// or perform mime type inference.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `data` does not contain a `data:` prefix
    /// - `data` is valid base64-encoded content
    pub fn from_parts(
        source_url: Option<Url>,
        mime_type: MediaType,
        data: String,
        detail: Option<Detail>,
        filename: Option<String>,
    ) -> Self {
        Self {
            source_url,
            mime_type,
            data,
            detail,
            filename,
        }
    }
}

impl std::fmt::Display for Base64File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

/// Custom deserializer for Base64File that validates the data field.
impl<'de> Deserialize<'de> for Base64File {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Base64FileHelper {
            #[serde(alias = "url")]
            source_url: Option<Url>,
            #[serde(default)]
            mime_type: Option<MediaType>,
            data: String,
            #[serde(default)]
            detail: Option<Detail>,
            #[serde(default)]
            filename: Option<String>,
        }

        let value = serde_json::Value::deserialize(deserializer)?;

        // Check if the deprecated "url" field is present (log warning)
        if value.get("url").is_some() && value.get("source_url").is_none() {
            crate::deprecation_warning(
                "`url` is deprecated for `Base64File`. Please use `source_url` instead.",
            );
        }

        let helper: Base64FileHelper =
            serde_json::from_value(value).map_err(serde::de::Error::custom)?;

        Base64File::new(
            helper.source_url,
            helper.mime_type,
            helper.data,
            helper.detail,
            helper.filename,
        )
        .map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Base64File {
    #[getter(url)]
    pub fn url_string(&self) -> Option<String> {
        self.source_url.as_ref().map(Url::to_string)
    }

    #[getter(mime_type)]
    pub fn mime_type_string(&self) -> String {
        self.mime_type.to_string()
    }

    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

/// Like `Base64File`, but without the data field.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Serialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct Base64FileMetadata {
    #[serde(alias = "url")] // DEPRECATED
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub source_url: Option<Url>,
    #[cfg_attr(feature = "ts-bindings", ts(type = "string"))]
    pub mime_type: MediaType,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub detail: Option<Detail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub filename: Option<String>,
}

impl<'de> Deserialize<'de> for Base64FileMetadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Base64FileMetadataHelper {
            #[serde(alias = "url")]
            source_url: Option<Url>,
            mime_type: MediaType,
            #[serde(default)]
            detail: Option<Detail>,
            #[serde(default)]
            filename: Option<String>,
        }

        let value = serde_json::Value::deserialize(deserializer)?;

        if value.get("url").is_some() && value.get("source_url").is_none() {
            crate::deprecation_warning(
                "`url` is deprecated for `Base64FileMetadata`. Please use `source_url` instead.",
            );
        }

        let helper: Base64FileMetadataHelper =
            serde_json::from_value(value).map_err(serde::de::Error::custom)?;

        Ok(Base64FileMetadata {
            source_url: helper.source_url,
            mime_type: helper.mime_type,
            detail: helper.detail,
            filename: helper.filename,
        })
    }
}

/// A file that can be located at a URL
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub struct UrlFile {
    #[schemars(with = "String")]
    pub url: Url,
    #[cfg_attr(feature = "ts-bindings", ts(type = "string | null"))]
    #[schemars(with = "Option<String>")]
    pub mime_type: Option<MediaType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub detail: Option<Detail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub filename: Option<String>,
}

impl<'de> Deserialize<'de> for UrlFile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct UrlFileHelper {
            url: Url,
            #[serde(default)]
            mime_type: Option<MediaType>,
            #[serde(default)]
            detail: Option<Detail>,
            #[serde(default)]
            filename: Option<String>,
        }

        let helper = UrlFileHelper::deserialize(deserializer)?;
        Ok(UrlFile {
            url: helper.url,
            mime_type: helper.mime_type,
            detail: helper.detail,
            filename: helper.filename,
        })
    }
}

/// A file stored in an object storage backend, without data.
/// This struct can be stored in the database. It's used by `StoredFile` (`StoredInput`).
/// Note: `File` supports both `ObjectStorageFilePointer` and `ObjectStorageFile`.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub struct ObjectStoragePointer {
    #[serde(alias = "url")] // DEPRECATED
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[schemars(with = "Option<String>")]
    pub source_url: Option<Url>,
    #[cfg_attr(feature = "ts-bindings", ts(type = "string"))]
    #[schemars(with = "String")]
    pub mime_type: MediaType,
    pub storage_path: StoragePath,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub detail: Option<Detail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub filename: Option<String>,
}

impl<'de> Deserialize<'de> for ObjectStoragePointer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ObjectStoragePointerHelper {
            #[serde(alias = "url")]
            source_url: Option<Url>,
            mime_type: MediaType,
            storage_path: StoragePath,
            #[serde(default)]
            detail: Option<Detail>,
            #[serde(default)]
            filename: Option<String>,
        }

        let value = serde_json::Value::deserialize(deserializer)?;

        if value.get("url").is_some() && value.get("source_url").is_none() {
            crate::deprecation_warning(
                "`url` is deprecated for `ObjectStoragePointer`. Please use `source_url` instead.",
            );
        }

        let helper: ObjectStoragePointerHelper =
            serde_json::from_value(value).map_err(serde::de::Error::custom)?;

        Ok(ObjectStoragePointer {
            source_url: helper.source_url,
            mime_type: helper.mime_type,
            storage_path: helper.storage_path,
            detail: helper.detail,
            filename: helper.filename,
        })
    }
}

/// A file stored in an object storage backend, with data.
/// This struct can NOT be stored in the database.
/// Note: `File` supports both `ObjectStorageFilePointer` and `ObjectStorageFile`.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[export_schema]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ObjectStorageFile {
    #[serde(flatten)]
    pub file: ObjectStoragePointer,
    pub data: String,
}

/// A file that we failed to read from object storage.
/// This struct can NOT be stored in the database.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub struct ObjectStorageError {
    #[serde(flatten)]
    pub file: ObjectStoragePointer,
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub error: Option<String>,
}

/// A file for an inference or a datapoint.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
#[serde(tag = "file_type", rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub enum File {
    #[schemars(title = "FileUrlFile")]
    Url(UrlFile),
    #[schemars(title = "FileBase64")]
    Base64(Base64File),
    #[schemars(title = "FileObjectStoragePointer")]
    ObjectStoragePointer(ObjectStoragePointer),
    #[schemars(title = "FileObjectStorage")]
    ObjectStorage(ObjectStorageFile),
    #[schemars(title = "FileObjectStorageError")]
    ObjectStorageError(ObjectStorageError),
}

/// Allow deserializing File as either tagged or untagged format.
/// This is a backwards compatibility feature for a while until we're confident that clients
/// are sending us the correct tagged versions.
impl<'de> Deserialize<'de> for File {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(tag = "file_type", rename_all = "snake_case")]
        enum TaggedFile {
            Url {
                url: Url,
                mime_type: Option<MediaType>,
                #[serde(default)]
                detail: Option<Detail>,
                #[serde(default)]
                filename: Option<String>,
            },
            Base64 {
                #[serde(default)]
                mime_type: Option<MediaType>,
                data: String,
                #[serde(default)]
                detail: Option<Detail>,
                #[serde(default)]
                filename: Option<String>,
            },
            ObjectStoragePointer {
                source_url: Option<Url>,
                mime_type: MediaType,
                storage_path: StoragePath,
                #[serde(default)]
                detail: Option<Detail>,
                #[serde(default)]
                filename: Option<String>,
            },
            ObjectStorage {
                #[serde(flatten)]
                file: ObjectStoragePointer,
                data: String,
            },
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum LegacyUntaggedFile {
            Url {
                url: Url,
                #[serde(default)]
                mime_type: Option<MediaType>,
                #[serde(default)]
                detail: Option<Detail>,
                #[serde(default)]
                filename: Option<String>,
            },
            Base64 {
                #[serde(default)]
                mime_type: Option<MediaType>,
                data: String,
                #[serde(default)]
                detail: Option<Detail>,
                #[serde(default)]
                filename: Option<String>,
            },
            ObjectStoragePointer {
                source_url: Option<Url>,
                mime_type: MediaType,
                storage_path: StoragePath,
                #[serde(default)]
                detail: Option<Detail>,
                #[serde(default)]
                filename: Option<String>,
            },
            ObjectStorage {
                #[serde(flatten)]
                file: ObjectStoragePointer,
                data: String,
            },
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum FileTaggedOrUntagged {
            Tagged(TaggedFile),
            Untagged(LegacyUntaggedFile),
        }

        match FileTaggedOrUntagged::deserialize(deserializer)? {
            FileTaggedOrUntagged::Tagged(TaggedFile::Url {
                url,
                mime_type,
                detail,
                filename,
            }) => Ok(File::Url(UrlFile {
                url,
                mime_type,
                detail,
                filename,
            })),
            FileTaggedOrUntagged::Tagged(TaggedFile::Base64 {
                mime_type,
                data,
                detail,
                filename,
            }) => Ok(File::Base64(
                Base64File::new(None, mime_type, data, detail, filename)
                    .map_err(serde::de::Error::custom)?,
            )),
            FileTaggedOrUntagged::Tagged(TaggedFile::ObjectStoragePointer {
                source_url,
                mime_type,
                storage_path,
                detail,
                filename,
            }) => Ok(File::ObjectStoragePointer(ObjectStoragePointer {
                source_url,
                mime_type,
                storage_path,
                detail,
                filename,
            })),
            FileTaggedOrUntagged::Tagged(TaggedFile::ObjectStorage { file, data }) => {
                Ok(File::ObjectStorage(ObjectStorageFile { file, data }))
            }
            FileTaggedOrUntagged::Untagged(LegacyUntaggedFile::Url {
                url,
                mime_type,
                detail,
                filename,
            }) => Ok(File::Url(UrlFile {
                url,
                mime_type,
                detail,
                filename,
            })),
            FileTaggedOrUntagged::Untagged(LegacyUntaggedFile::Base64 {
                mime_type,
                data,
                detail,
                filename,
            }) => Ok(File::Base64(
                Base64File::new(None, mime_type, data, detail, filename)
                    .map_err(serde::de::Error::custom)?,
            )),
            FileTaggedOrUntagged::Untagged(LegacyUntaggedFile::ObjectStoragePointer {
                source_url,
                mime_type,
                storage_path,
                detail,
                filename,
            }) => Ok(File::ObjectStoragePointer(ObjectStoragePointer {
                source_url,
                mime_type,
                storage_path,
                detail,
                filename,
            })),
            FileTaggedOrUntagged::Untagged(LegacyUntaggedFile::ObjectStorage { file, data }) => {
                Ok(File::ObjectStorage(ObjectStorageFile { file, data }))
            }
        }
    }
}
