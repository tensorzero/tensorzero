use object_store::path::Path;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{Error, ErrorDetails};

use super::{file::mime_type_to_ext, Base64File};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use tensorzero_derive::export_schema;

/// Configuration for the object storage backend
/// Currently, we only support S3-compatible object storage and local filesystem storage
/// We test against Amazon S3, GCS, Cloudflare R2, and Minio
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
#[derive(ts_rs::TS)]
#[export_schema]
#[ts(export)]
pub enum StorageKind {
    #[schemars(title = "StorageKindS3Compatible")]
    S3Compatible {
        // TODO(shuyangli): mark all of these as ts(optional)
        bucket_name: Option<String>,
        region: Option<String>,
        endpoint: Option<String>,
        allow_http: Option<bool>,
        /// An extra prefix to prepend to the object key.
        /// This is only enabled in e2e tests, to prevent clashes between concurrent test runs.
        #[cfg(feature = "e2e_tests")]
        #[ts(skip)]
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

impl StorageKind {
    /// Get the extra prefix for the object key during e2e-tests
    #[cfg(feature = "e2e_tests")]
    fn prefix(&self) -> &str {
        match self {
            StorageKind::S3Compatible { prefix, .. } => prefix,
            _ => "",
        }
    }

    /// During a normal run, we never use a prefix on the object key.
    /// See `StorageKind::S3Compatible.prefix`
    #[cfg(not(feature = "e2e_tests"))]
    #[expect(clippy::unused_self)]
    fn prefix(&self) -> &str {
        ""
    }
    pub fn file_path(self, image: &Base64File) -> Result<StoragePath, Error> {
        // Compute a content-addressed hash from the file data only.
        // Intentionally excludes the `detail` field so that the same file with different
        // detail settings (low/high/auto) will map to the same storage location for deduplication.
        let hash = blake3::hash(image.data().as_bytes());
        // This is a best-effort attempt to get a suffix in the object-store path, to make things
        // nicer for people browsing the object store.
        // None of our code depends on this file extension being correct, as we store the original
        // mime-type in our database, and use it in the ui when rendering a preview of the file.
        let suffix = mime_type_to_ext(&image.mime_type)?
            .map(|s| format!(".{s}"))
            .unwrap_or_default();
        let path = Path::parse(format!(
            "{}observability/files/{hash}{suffix}",
            self.prefix()
        ))
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to construct object_store path: {e}"),
            })
        })?;
        Ok(StoragePath { kind: self, path })
    }
}

/// Path to a file in an object storage backend.
/// This is part of the public API for `File`s. In particular, this is useful for roundtripping
/// unresolved inputs from stored inferences or datapoints, without requiring clients to fetch
/// file data first.
#[derive(ts_rs::TS, Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[ts(export)]
#[export_schema]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct StoragePath {
    pub kind: StorageKind,
    #[serde(
        serialize_with = "serialize_storage_path",
        deserialize_with = "deserialize_storage_path"
    )]
    #[ts(type = "string")]
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
    object_store::path::Path::parse(&path).map_err(serde::de::Error::custom)
}
