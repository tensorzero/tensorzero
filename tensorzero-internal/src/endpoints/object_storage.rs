use std::borrow::Cow;

use axum::{
    extract::{Query, State},
    Json,
};
use serde::{Deserialize, Serialize};

use crate::{
    config_parser::ObjectStoreInfo,
    error::{Error, ErrorDetails},
    gateway_util::{AppState, AppStateData},
    inference::types::storage::StoragePath,
};
use aws_smithy_types::base64;

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "e2e_tests", derive(PartialEq))]
pub struct ObjectResponse {
    pub data: String,
    pub reused_object_store: bool,
}

#[derive(Debug, Deserialize)]
pub struct PathParams {
    pub storage_path: String,
}

/// Fetches an object using the object store and path specified by the encoded `StoragePath`.
/// This does not need to match the gateway's current object store (e.g. a `StorageKind::Filesystem`)
/// could be provided even if the gateway is configured with `StorageKind::S3Compatible`.
/// However, if the provider requires authentication, the gateway must have the correct credentials
/// set as environment variables.
///
/// This is invoked as `GET /internal/object_storage?storage_path=<urlencoded_storagepath>`.
/// The `<urlencoded_storagepath>` value constructed by serializing `StoragePath` to JSON,
/// and the urlencoding the resulting string.
/// For example, `GET /internal/object_storage?storage_path={%22kind%22:{%22type%22:%22filesystem%22,%22path%22:%22/tmp%22},%22path%22:%22fake-tensorzero-file%22}`
pub async fn get_object_handler(
    State(AppStateData { config, .. }): AppState,
    Query(params): Query<PathParams>,
) -> Result<Json<ObjectResponse>, Error> {
    let storage_path: StoragePath = serde_json::from_str(&params.storage_path).map_err(|e| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Error parsing storage path: {}", e),
        })
    })?;
    // Use the existing object store if it matches the requested kind, so
    // that we can re-use our connection pool.
    let store = match &config.object_store_info {
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
                message: format!("Error getting object: {}", e),
            })
        })?;
    let bytes = object.bytes().await.map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Error getting object bytes: {}", e),
        })
    })?;
    Ok(Json(ObjectResponse {
        data: base64::encode(&bytes),
        reused_object_store: matches!(store, Cow::Borrowed(_)),
    }))
}
