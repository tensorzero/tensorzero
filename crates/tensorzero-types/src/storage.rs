//! Storage types for object storage paths.
//!
//! This module contains types for representing object storage locations.

use object_store::path::Path;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tensorzero_derive::export_schema;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// Configuration for the object storage backend
/// Currently, we only support S3-compatible object storage and local filesystem storage
/// We test against Amazon S3, GCS, Cloudflare R2, and Minio
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[export_schema]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum StorageKind {
    #[schemars(title = "StorageKindS3Compatible")]
    S3Compatible {
        bucket_name: Option<String>,
        region: Option<String>,
        endpoint: Option<String>,
        allow_http: Option<bool>,
        /// An extra prefix to prepend to the object key.
        /// This is only enabled in e2e tests, to prevent clashes between concurrent test runs.
        #[cfg(feature = "e2e_tests")]
        #[cfg_attr(feature = "ts-bindings", ts(skip))]
        #[serde(default)]
        prefix: String,
    },
    #[schemars(title = "StorageKindFilesystem")]
    Filesystem { path: String },
    // This must be set explicitly in `tensorzero.toml` to allow image requests to succeed
    // By default, requests will fail (we'll have a `None` for the outer `ObjectStoreData`)
    #[schemars(title = "StorageKindDisabled")]
    Disabled,
}

/// Path to a file in an object storage backend.
/// This is part of the public API for `File`s. In particular, this is useful for roundtripping
/// unresolved inputs from stored inferences or datapoints, without requiring clients to fetch
/// file data first.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct StoragePath {
    pub kind: StorageKind,
    #[serde(
        serialize_with = "serialize_storage_path",
        deserialize_with = "deserialize_storage_path"
    )]
    #[cfg_attr(feature = "ts-bindings", ts(type = "string"))]
    #[schemars(with = "String")]
    pub path: object_store::path::Path,
}

impl std::fmt::Display for StoragePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl StoragePath {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

fn serialize_storage_path<S: Serializer>(
    path: &object_store::path::Path,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    path.to_string().serialize(serializer)
}

fn deserialize_storage_path<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<object_store::path::Path, D::Error> {
    let path = String::deserialize(deserializer)?;
    Path::parse(&path).map_err(serde::de::Error::custom)
}
