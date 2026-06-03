use std::borrow::Cow;

use axum::{
    Json,
    extract::{Query, State},
};
use serde::{Deserialize, Serialize};

use crate::{
    config::ObjectStoreInfo,
    error::{Error, ErrorDetails},
    inference::types::storage::StoragePath,
    utils::gateway::{AppState, AppStateData},
};
use aws_smithy_types::base64;
use object_store::ObjectStoreExt;

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "e2e_tests", derive(PartialEq))]
pub struct ObjectResponse {
    pub data: String,
}

#[derive(Debug, Deserialize)]
pub struct PathParams {
    pub path: String,
}

/// Fetches an object from the gateway's configured object store by path.
///
/// This is invoked as `GET /internal/object_storage?path=<urlencoded_path>`.
/// `path` is the object's key inside the configured store — the gateway does **not** accept
/// caller-supplied filesystem roots, S3 endpoints, buckets, regions, or `allow_http` flags,
/// so the endpoint cannot be coerced into opening a store other than the configured one.
/// In-process callers that need to resolve a `StoragePath` recorded under a different
/// configuration (e.g. after a `[object_storage]` migration) can call `get_object` directly.
pub async fn get_object_handler(
    State(AppStateData { config, .. }): AppState,
    Query(params): Query<PathParams>,
) -> Result<Json<ObjectResponse>, Error> {
    Ok(Json(
        fetch_object_by_path(config.object_store_info.as_ref(), &params.path).await?,
    ))
}

/// Parses a caller-supplied path string and fetches it from the configured store.
///
/// `object_store::path::Path::parse` rejects relative segments (`.`, `..`), empty
/// segments, and ASCII control characters, so the resulting filesystem read is
/// guaranteed to land inside the configured `[object_storage]` root and cannot
/// traverse out of it.
pub async fn fetch_object_by_path(
    object_store_info: Option<&ObjectStoreInfo>,
    path: &str,
) -> Result<ObjectResponse, Error> {
    let parsed = object_store::path::Path::parse(path).map_err(|e| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Error parsing object path: {e}"),
        })
    })?;
    fetch_from_configured_store(object_store_info, &parsed).await
}

/// Fetches an object by path from the gateway's configured object store. Used by the
/// `/internal/object_storage` HTTP handler and by the `Client::get_object` SDK method —
/// neither lets the caller pick a different store, so this is the security-relevant
/// fetch path for untrusted callers.
pub async fn fetch_from_configured_store(
    object_store_info: Option<&ObjectStoreInfo>,
    path: &object_store::path::Path,
) -> Result<ObjectResponse, Error> {
    let store = object_store_info
        .and_then(|info| info.object_store.as_ref())
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidRequest {
                message:
                    "Object storage is not configured on this gateway, or is set to `disabled`. \
                     Set `[object_storage]` in the gateway config to use this endpoint."
                        .to_string(),
            })
        })?;
    let object = store.get(path).await.map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Error getting object: {e}"),
        })
    })?;
    let bytes = object.bytes().await.map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Error getting object bytes: {e}"),
        })
    })?;
    Ok(ObjectResponse {
        data: base64::encode(&bytes),
    })
}

/// Fetches an object using the object store and path specified by the encoded `StoragePath`.
/// This does not need to match the gateway's current object store (e.g. a `StorageKind::Filesystem`
/// could be provided even if the gateway is configured with `StorageKind::S3Compatible`).
/// However, if the provider requires authentication, the gateway must have the correct credentials
/// set as environment variables.
///
/// SECURITY: this function trusts its `storage_path` argument and will build an object store
/// from a caller-supplied filesystem root or S3 endpoint when the kind does not match the
/// configured store. It must NOT be exposed to untrusted callers — the HTTP handler and SDK
/// surface go through `fetch_from_configured_store` instead. This entry point exists only for
/// in-process code resolving `StoragePath`s the gateway itself recorded under a different
/// configuration.
pub async fn get_object(
    object_store_info: Option<&ObjectStoreInfo>,
    storage_path: StoragePath,
) -> Result<ObjectResponse, Error> {
    // Use the existing object store if it matches the requested kind, so
    // that we can re-use our connection pool.
    let store = match object_store_info {
        Some(store) if store.kind == storage_path.kind => Cow::Borrowed(store),
        _ => Cow::Owned(
            ObjectStoreInfo::new(Some(storage_path.kind))?.ok_or_else(|| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: "Could not create ObjectStoreInfo from provided `kind`".to_string(),
                })
            })?,
        ),
    };
    let object = store
        .object_store
        .as_ref()
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidRequest {
                message: "Object store was disabled".to_string(),
            })
        })?
        .get(&storage_path.path)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Error getting object: {e}"),
            })
        })?;
    let bytes = object.bytes().await.map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Error getting object bytes: {e}"),
        })
    })?;
    Ok(ObjectResponse {
        data: base64::encode(&bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::storage::StorageKind;
    use aws_smithy_types::base64::decode as base64_decode;
    use googletest::prelude::*;
    use std::fs;
    use tempfile::TempDir;

    fn make_filesystem_store(prefix: &std::path::Path) -> ObjectStoreInfo {
        ObjectStoreInfo::new(Some(StorageKind::Filesystem {
            path: prefix.to_string_lossy().into_owned(),
        }))
        .expect("filesystem store construction should not fail")
        .expect("filesystem store should be configured")
    }

    #[gtest]
    #[tokio::test]
    async fn rejects_when_object_storage_unconfigured() {
        let err = fetch_object_by_path(None, "any/file")
            .await
            .expect_err("unconfigured object storage should reject the request");
        let msg = err.to_string();
        assert!(
            msg.contains("Object storage is not configured"),
            "unexpected error: {msg}"
        );
        assert_eq!(err.status_code(), http::StatusCode::BAD_REQUEST);
    }

    /// The handler must not allow path-traversal segments to escape the configured root.
    /// `object_store::path::Path::parse` rejects `.`/`..`/empty segments at parse time,
    /// so the security guarantee is enforced before any filesystem call happens.
    #[gtest]
    #[tokio::test]
    async fn rejects_parent_directory_traversal_in_filesystem_mode() {
        // Lay out:
        //   <root>/
        //     storage/        <- configured `[object_storage]` root
        //       inside.txt    <- should be readable
        //     private-file    <- MUST NOT be readable via the endpoint
        let root = TempDir::new().expect("create temp root");
        let storage_dir = root.path().join("storage");
        fs::create_dir(&storage_dir).expect("create storage dir");
        fs::write(storage_dir.join("inside.txt"), b"safe-content").expect("write inside file");
        fs::write(root.path().join("private-file"), b"super-secret")
            .expect("write private file outside prefix");

        let store = make_filesystem_store(&storage_dir);

        // Sanity check: the file inside the prefix is reachable through the endpoint.
        let inside = fetch_object_by_path(Some(&store), "inside.txt")
            .await
            .expect("file inside configured root should be readable");
        let decoded = base64_decode(inside.data).expect("base64 decodes");
        assert_eq!(
            decoded, b"safe-content",
            "expected to read the file inside the configured root"
        );

        // Each of these tries to break out of `storage/` and read `private-file` (or
        // `/etc/passwd` etc.) on the host filesystem. They must all fail before the
        // filesystem read happens — i.e. `Path::parse` rejects the input.
        let traversal_attempts = [
            "../private-file",
            "../../etc/passwd",
            "foo/../../private-file",
            "./private-file",
            "foo/./bar",
            "foo//bar",
        ];
        for attempt in traversal_attempts {
            let result = fetch_object_by_path(Some(&store), attempt).await;
            let Err(err) = result else {
                panic!(
                    "expected path {attempt:?} to be rejected at parse time, but the fetch \
                     succeeded — this indicates path traversal is possible"
                );
            };
            let msg = err.to_string();
            assert!(
                msg.contains("Error parsing object path"),
                "path {attempt:?} should be rejected by Path::parse, got error: {msg}"
            );
            assert_eq!(
                err.status_code(),
                http::StatusCode::BAD_REQUEST,
                "path {attempt:?} should produce a 400, got {}",
                err.status_code()
            );
            // The private file's content must never appear in an error message.
            assert!(
                !msg.contains("super-secret"),
                "error message for {attempt:?} unexpectedly leaks file contents: {msg}"
            );
        }
    }

    /// An "absolute-looking" path like `/etc/passwd` is normalized by `Path::parse`
    /// (which strips the leading `/`), so it resolves under the configured prefix —
    /// it does NOT reach the real `/etc/passwd`.
    #[gtest]
    #[tokio::test]
    async fn absolute_looking_paths_resolve_inside_prefix() {
        let root = TempDir::new().expect("create temp root");
        fs::write(root.path().join("private-file"), b"super-secret").expect("write private file");
        let storage_dir = root.path().join("storage");
        fs::create_dir(&storage_dir).expect("create storage dir");
        let store = make_filesystem_store(&storage_dir);

        // `/etc/passwd` parses as the relative segments `etc/passwd` and is looked up
        // under the prefix — it does NOT escape to `/etc/passwd` on the host. Since
        // `<prefix>/etc/passwd` does not exist, this returns "not found".
        let err = fetch_object_by_path(Some(&store), "/etc/passwd")
            .await
            .expect_err("nonexistent path under prefix should error");
        let msg = err.to_string();
        assert!(
            msg.contains("Error getting object"),
            "expected `not found` style error from the store, got: {msg}"
        );
        // And under no circumstances should the request reach the file outside the prefix.
        assert!(
            !msg.contains("super-secret"),
            "error unexpectedly leaks file contents: {msg}"
        );
    }

    #[gtest]
    #[tokio::test]
    async fn missing_object_inside_prefix_returns_not_found_error() {
        let root = TempDir::new().expect("create temp root");
        let store = make_filesystem_store(root.path());
        let err = fetch_object_by_path(Some(&store), "does/not/exist")
            .await
            .expect_err("missing file should error");
        assert!(
            err.to_string().contains("Error getting object"),
            "unexpected error: {err}"
        );
    }
}
