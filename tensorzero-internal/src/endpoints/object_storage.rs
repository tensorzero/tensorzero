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
pub struct ObjectResponse {
    data: String,
}

#[derive(Debug, Deserialize)]
pub struct PathParams {
    storage_path: String,
}

pub async fn get_object_handler(
    State(AppStateData { .. }): AppState,
    Query(params): Query<PathParams>,
) -> Result<Json<ObjectResponse>, Error> {
    // TODO - should we re-use the config object store if it matches?
    let storage_path: StoragePath = serde_json::from_str(&params.storage_path).map_err(|e| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Error parsing storage path: {}", e),
        })
    })?;
    let store = ObjectStoreInfo::new(Some(storage_path.kind))?.ok_or_else(|| {
        Error::new(ErrorDetails::InvalidRequest {
            message: "Could not create ObjectStoreInfo from provided `kind`".to_string(),
        })
    })?;
    let object = store
        .object_store
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
    }))
}
