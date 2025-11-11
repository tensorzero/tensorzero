use axum::{
    body::Body,
    extract::Path,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use crate::{apply_delay, error::Error};

// Storage for GCS objects (bucket/path -> content)
static GCS_OBJECTS: OnceLock<Mutex<HashMap<String, Vec<u8>>>> = OnceLock::new();

/// PUT /gcs/{bucket}/{*path} - Upload object to GCS
pub async fn upload_object(
    Path((bucket, path)): Path<(String, String)>,
    body: Body,
) -> Result<Response, Error> {
    apply_delay().await;

    // Read the body
    let bytes = axum::body::to_bytes(body, usize::MAX)
        .await
        .map_err(|e| Error::new(format!("Failed to read body: {}", e), StatusCode::BAD_REQUEST))?;

    let key = format!("{}/{}", bucket, path);
    let mut objects = GCS_OBJECTS.get_or_init(Default::default).lock().unwrap();
    objects.insert(key.clone(), bytes.to_vec());

    tracing::debug!("Stored GCS object: {}", key);

    Ok(Response::builder()
        .status(StatusCode::OK)
        .body(Body::empty())
        .unwrap())
}

/// GET /gcs/{bucket}/{*path} - Download object from GCS
pub async fn download_object(
    Path((bucket, path)): Path<(String, String)>,
) -> Result<Response, Error> {
    apply_delay().await;

    let key = format!("{}/{}", bucket, path);
    let objects = GCS_OBJECTS.get_or_init(Default::default).lock().unwrap();

    let data = objects.get(&key).ok_or_else(|| {
        Error::new(
            format!("Object not found: {}", key),
            StatusCode::NOT_FOUND,
        )
    })?;

    tracing::debug!("Retrieved GCS object: {} ({} bytes)", key, data.len());

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(data.clone()))
        .unwrap())
}
