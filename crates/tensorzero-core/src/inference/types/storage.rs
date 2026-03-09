use object_store::path::Path;

use crate::error::{Error, ErrorDetails};

use super::{Base64File, file::mime_type_to_ext};

// Re-export wire types from tensorzero-types
pub use tensorzero_types::{StorageKind, StoragePath};

/// Extension trait for StorageKind operations that require tensorzero-core dependencies.
pub trait StorageKindExt {
    /// Get the extra prefix for the object key during e2e-tests
    fn prefix(&self) -> &str;

    /// Compute the storage path for a file based on its content.
    fn file_path(self, image: &Base64File) -> Result<StoragePath, Error>;
}

impl StorageKindExt for StorageKind {
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
    fn prefix(&self) -> &'static str {
        // self is intentionally unused in non-e2e builds
        let _ = self;
        ""
    }

    fn file_path(self, image: &Base64File) -> Result<StoragePath, Error> {
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
