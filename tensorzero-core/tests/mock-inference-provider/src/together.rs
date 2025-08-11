use std::{
    collections::{hash_map::Entry, HashMap},
    sync::{Mutex, OnceLock},
};

use anyhow::anyhow;
use axum::http::StatusCode;
use axum::{
    extract::{Json, Multipart, Path},
    response::IntoResponse,
};
use rand::distr::{Alphanumeric, SampleString};
use serde::Deserialize;

pub struct TogetherError(anyhow::Error);

impl<E> From<E> for TogetherError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

impl IntoResponse for TogetherError {
    fn into_response(self) -> axum::response::Response {
        let status = axum::http::StatusCode::INTERNAL_SERVER_ERROR;
        let body = format!("Internal Server Error: {}", self.0);
        (status, body).into_response()
    }
}

type Result<T> = std::result::Result<T, TogetherError>;

static TOGETHER_FINE_TUNING_JOBS: OnceLock<Mutex<HashMap<String, FineTuningJob>>> = OnceLock::new();
static TOGETHER_FILES: OnceLock<Mutex<HashMap<String, File>>> = OnceLock::new();

struct FineTuningJob {
    poll_count: usize,
    model: String,
}

#[derive(Deserialize)]
struct CreateFineTuneRequest {
    training_file: String,
    #[allow(dead_code)]
    validation_file: Option<String>,
    model: String,
    #[allow(dead_code)]
    n_epochs: Option<u32>,
    #[allow(dead_code)]
    n_checkpoints: Option<u32>,
    #[allow(dead_code)]
    n_evals: Option<u32>,
    #[allow(dead_code)]
    warmup_ratio: Option<f64>,
    #[allow(dead_code)]
    max_grad_norm: Option<f64>,
    #[allow(dead_code)]
    weight_decay: Option<f64>,
    #[allow(dead_code)]
    batch_size: u32,
    #[allow(dead_code)]
    learning_rate: f64,
    #[allow(dead_code)]
    lr_scheduler: serde_json::Value,
    #[allow(dead_code)]
    training_method: serde_json::Value,
    #[allow(dead_code)]
    training_type: serde_json::Value,
}

pub async fn create_fine_tune(
    Json(body): Json<CreateFineTuneRequest>,
) -> Result<Json<serde_json::Value>> {
    let job_id = format!(
        "mock-together-finetune-{}",
        Alphanumeric.sample_string(&mut rand::rng(), 10)
    );
    
    // Validate that the training file exists
    let files = TOGETHER_FILES.get().unwrap();
    let files_guard = files.lock().unwrap();
    if !files_guard.contains_key(&body.training_file) {
        return Err(TogetherError::from(anyhow!("Training file not found")));
    }
    
    TOGETHER_FINE_TUNING_JOBS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .insert(
            job_id.clone(),
            FineTuningJob {
                poll_count: 0,
                model: body.model,
            },
        );

    Ok(Json(serde_json::json!({
        "id": job_id,
        "status": "pending"
    })))
}

#[derive(Deserialize)]
pub struct FineTuneParams {
    job_id: String,
}

const POLLS_UNTIL_READY: usize = 2;

pub async fn get_fine_tune(Path(params): Path<FineTuneParams>) -> Result<Json<serde_json::Value>> {
    let mut jobs = TOGETHER_FINE_TUNING_JOBS
        .get_or_init(Default::default)
        .lock()
        .unwrap();
    let job = jobs
        .get_mut(&params.job_id)
        .ok_or_else(|| anyhow!("Job not found"))?;

    job.poll_count += 1;
    if job.poll_count < POLLS_UNTIL_READY {
        Ok(Json(serde_json::json!({
            "id": params.job_id,
            "status": "running",
            "token_count": 1000000
        })))
    } else {
        if params.job_id.contains("error") {
            return Ok(Json(serde_json::json!({
                "id": params.job_id,
                "status": "error",
                "token_count": 1000000,
                "error": "Model error"
            })));
        }
        Ok(Json(serde_json::json!({
            "id": params.job_id,
            "status": "completed",
            "model_output_name": format!("{}-fine-tuned", job.model),
            "token_count": 1000000
        })))
    }
}

struct File {
    purpose: String,
}

pub async fn upload_file(mut form: Multipart) -> Result<Json<serde_json::Value>> {
    let file_id = format!(
        "mock-together-file-{}",
        Alphanumeric.sample_string(&mut rand::rng(), 10)
    );

    let mut file_len = None;
    let mut purpose = None;
    while let Some(field) = form.next_field().await.unwrap() {
        if field.name() == Some("file") {
            let bytes = field.bytes().await.unwrap();
            file_len = Some(bytes.len());
        } else if field.name() == Some("purpose") {
            purpose = Some(field.text().await.unwrap());
        }
    }

    if let Some(purpose) = &purpose {
        TOGETHER_FILES
            .get_or_init(Default::default)
            .lock()
            .unwrap()
            .insert(
                file_id.clone(),
                File {
                    purpose: purpose.clone(),
                },
            );
    }

    Ok(Json(serde_json::json!({
        "id": file_id,
        "object": "file",
        "bytes": file_len,
        "purpose": purpose,
    })))
}