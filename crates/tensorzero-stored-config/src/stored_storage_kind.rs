use serde::{Deserialize, Serialize};
use tensorzero_types::StorageKind;

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredStorageKind {
    S3Compatible {
        bucket_name: Option<String>,
        region: Option<String>,
        endpoint: Option<String>,
        allow_http: Option<bool>,
        // The prefix field behind `e2e_tests` feature is never stored in toml and only used
        // directly in e2e tests, so we skip it.
    },
    Filesystem {
        path: String,
    },
    Disabled,
}

impl From<StoredStorageKind> for StorageKind {
    fn from(stored: StoredStorageKind) -> Self {
        match stored {
            StoredStorageKind::S3Compatible {
                bucket_name,
                region,
                endpoint,
                allow_http,
            } => StorageKind::S3Compatible {
                bucket_name,
                region,
                endpoint,
                allow_http,
                #[cfg(feature = "e2e_tests")]
                prefix: String::new(),
            },
            StoredStorageKind::Filesystem { path } => StorageKind::Filesystem { path },
            StoredStorageKind::Disabled => StorageKind::Disabled,
        }
    }
}
