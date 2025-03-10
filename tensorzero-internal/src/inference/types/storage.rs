use object_store::path::Path;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{Error, ErrorDetails};

use super::{Base64Image, ImageKind};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StorageKind {
    S3Compatible {
        bucket_name: Option<String>,
        region: Option<String>,
        endpoint: Option<String>,
        allow_http: Option<bool>,
        /// An extra prefix to prepend to the object key.
        /// This is only enabled in e2e tests, to prevent clashes between concurrent test runs.
        #[cfg(feature = "e2e_tests")]
        #[serde(default)]
        prefix: String,
    },
    Filesystem {
        path: String,
    },
    // This must be set explicitly in `tensorzero.toml` to allow image requests to succeed
    // By default, requests will fail (we'll have a `None` for the outer `ObjectStoreData`)
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
    fn prefix(&self) -> &str {
        ""
    }
    pub fn image_path(self, image: &Base64Image) -> Result<StoragePath, Error> {
        let hash = blake3::hash(
            image
                .data
                .as_ref()
                .ok_or_else(|| {
                    Error::new(ErrorDetails::InternalError {
                        message: "Image data should have been present in `StorageKind.image_path`"
                            .to_string(),
                    })
                })?
                .as_bytes(),
        );
        let suffix = match image.mime_type {
            ImageKind::Jpeg => "jpg",
            ImageKind::Png => "png",
            ImageKind::WebP => "webp",
        };
        let path = Path::parse(format!(
            "{}observability/images/{hash}.{suffix}",
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoragePath {
    pub kind: StorageKind,
    #[serde(
        serialize_with = "serialize_storage_path",
        deserialize_with = "deserialize_storage_path"
    )]
    pub path: object_store::path::Path,
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
