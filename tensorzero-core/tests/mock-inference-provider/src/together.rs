use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use anyhow::anyhow;
use axum::{
    extract::{Multipart, Path},
    response::IntoResponse,
    Json,
};
use rand::distr::{Alphanumeric, SampleString};
use serde::{Deserialize, Serialize};

use crate::apply_delay;

// Mirror the TogetherBatchSize enum from tensorzero-core
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TogetherBatchSize {
    Number(u32),
    Description(TogetherBatchSizeDescription),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TogetherBatchSizeDescription {
    Max,
}

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

static TOGETHER_FINE_TUNING_JOBS: OnceLock<Mutex<HashMap<String, TogetherFineTuningJob>>> =
    OnceLock::new();

struct TogetherFineTuningJob {
    poll_count: usize,
}

const POLLS_UNTIL_READY: usize = 2;

#[derive(Debug, Deserialize)]
#[expect(dead_code)]
pub struct TogetherCreateJobRequest {
    pub training_file: String,
    pub validation_file: Option<String>,
    pub model: String,
    pub n_epochs: Option<u32>,
    pub n_checkpoints: Option<u32>,
    pub n_evals: Option<u32>,
    pub warmup_ratio: Option<f64>,
    pub max_grad_norm: Option<f64>,
    pub weight_decay: Option<f64>,
    pub batch_size: TogetherBatchSize,
    pub lr_scheduler: serde_json::Value,
    pub learning_rate: f64,
    pub training_method: serde_json::Value,
    pub training_type: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct TogetherFileResponse {
    pub id: String,
}

#[derive(Debug, Serialize)]
pub struct TogetherCreateJobResponse {
    pub id: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
#[expect(dead_code)]
pub enum TogetherJobStatus {
    Pending,
    Queued,
    Running,
    Compressing,
    Uploading,
    CancelRequested,
    UserError,
    Cancelled,
    Error,
    Completed,
}

#[derive(Debug, Serialize)]
pub struct TogetherJobResponse {
    pub status: TogetherJobStatus,
    pub token_count: Option<u64>,
    pub model_output_name: Option<String>,
}

/// Handler for uploading files to Together
pub async fn upload_file(mut multipart: Multipart) -> Result<Json<TogetherFileResponse>> {
    apply_delay().await;
    // Process the multipart form data
    // Together sends three fields: file, purpose, and file_name
    let mut file_content = None;
    let mut _purpose = None;
    let mut _file_name = None;

    while let Some(field) = multipart.next_field().await? {
        let field_name = field.name().map(std::string::ToString::to_string);

        match field_name.as_deref() {
            Some("file") => {
                file_content = Some(field.bytes().await?);
            }
            Some("purpose") => {
                _purpose = Some(field.text().await?);
            }
            Some("file_name") => {
                _file_name = Some(field.text().await?);
            }
            _ => {
                // Ignore unknown fields
            }
        }
    }

    // Validate that we got the required fields
    if file_content.is_none() {
        return Err(anyhow!("Missing 'file' field in multipart form").into());
    }

    // Generate a random file ID
    let file_id = format!("file-{}", Alphanumeric.sample_string(&mut rand::rng(), 16));

    Ok(Json(TogetherFileResponse { id: file_id }))
}

/// Handler for creating a fine-tuning job
pub async fn create_fine_tuning_job(
    Json(_request): Json<TogetherCreateJobRequest>,
) -> Result<Json<TogetherCreateJobResponse>> {
    apply_delay().await;
    // Generate a random job ID
    let job_id = format!("ft-{}", Alphanumeric.sample_string(&mut rand::rng(), 16));

    TOGETHER_FINE_TUNING_JOBS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .insert(job_id.clone(), TogetherFineTuningJob { poll_count: 0 });

    Ok(Json(TogetherCreateJobResponse { id: job_id }))
}

/// Handler for getting the status of a fine-tuning job
pub async fn get_fine_tuning_job(Path(job_id): Path<String>) -> Result<Json<TogetherJobResponse>> {
    apply_delay().await;
    let mut jobs = TOGETHER_FINE_TUNING_JOBS
        .get_or_init(Default::default)
        .lock()
        .unwrap();

    let job = jobs
        .get_mut(&job_id)
        .ok_or_else(|| anyhow!("Job not found"))?;

    job.poll_count += 1;

    // Update status based on the number of polls
    let status = if job.poll_count < POLLS_UNTIL_READY {
        if job.poll_count == 1 {
            TogetherJobStatus::Queued
        } else {
            TogetherJobStatus::Running
        }
    } else {
        TogetherJobStatus::Completed
    };

    // Set token count and model output name when completed
    let (token_count, model_output_name) = if matches!(status, TogetherJobStatus::Completed) {
        (Some(1337), Some(format!("{job_id}-output")))
    } else {
        (None, None)
    };

    Ok(Json(TogetherJobResponse {
        status,
        token_count,
        model_output_name,
    }))
}
