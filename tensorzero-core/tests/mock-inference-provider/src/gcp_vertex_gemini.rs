use std::{
    collections::{hash_map::Entry, HashMap},
    sync::{Mutex, OnceLock},
};

use anyhow::anyhow;
use axum::http::StatusCode;
use axum::{
    extract::{Json, Path},
    response::IntoResponse,
};
use rand::distr::{Alphanumeric, SampleString};
use serde::Deserialize;
use serde_json::Value;

pub struct GCPVertexGeminiError(anyhow::Error);

impl<E> From<E> for GCPVertexGeminiError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

impl IntoResponse for GCPVertexGeminiError {
    fn into_response(self) -> axum::response::Response {
        let status = axum::http::StatusCode::INTERNAL_SERVER_ERROR;
        let body = format!("Internal Server Error: {}", self.0);
        (status, body).into_response()
    }
}

type Result<T> = std::result::Result<T, GCPVertexGeminiError>;

static GCP_VERTEX_GEMINI_FINE_TUNING_JOBS: OnceLock<Mutex<HashMap<String, FineTuningJob>>> = OnceLock::new();

struct FineTuningJob {
    poll_count: usize,
    model: String,
}

#[derive(Deserialize)]
struct CreateFineTuningRequest {
    #[allow(dead_code)]
    base_model: String,
    #[allow(dead_code)]
    supervised_tuning_spec: Value,
    #[allow(dead_code)]
    tuned_model_display_name: Option<String>,
    #[allow(dead_code)]
    service_account: Option<String>,
    #[allow(dead_code)]
    encryption_spec: Option<Value>,
}

pub async fn create_fine_tune(
    _body: Json<CreateFineTuningRequest>,
) -> Result<Json<serde_json::Value>> {
    let job_id = format!(
        "mock-gcp-vertex-gemini-finetune-{}",
        Alphanumeric.sample_string(&mut rand::rng(), 10)
    );

    // For mocking purposes, we'll just generate a mock job URL
    let job_url = format!(
        "projects/mock-project/locations/mock-region/tuningJobs/{}",
        job_id
    );

    GCP_VERTEX_GEMINI_FINE_TUNING_JOBS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .insert(
            job_id.clone(),
            FineTuningJob {
                poll_count: 0,
                model: "gemini-2.0-flash-lite-001".to_string(),
            },
        );

    Ok(Json(serde_json::json!({
        "name": job_url,
        "state": "JOB_STATE_PENDING"
    })))
}

#[derive(Deserialize)]
pub struct FineTuneJobPath {
    project_id: String,
    region: String,
    job_id: String,
}

const POLLS_UNTIL_READY: usize = 2;

pub async fn get_fine_tune_job(Path(params): Path<FineTuneJobPath>) -> Result<Json<serde_json::Value>> {
    let mut jobs = GCP_VERTEX_GEMINI_FINE_TUNING_JOBS
        .get_or_init(Default::default)
        .lock()
        .unwrap();
    let job = jobs
        .get_mut(&params.job_id)
        .ok_or_else(|| anyhow!("Job not found"))?;

    job.poll_count += 1;
    if job.poll_count < POLLS_UNTIL_READY {
        Ok(Json(serde_json::json!({
            "name": format!("projects/{}/locations/{}/tuningJobs/{}", params.project_id, params.region, params.job_id),
            "state": "JOB_STATE_RUNNING",
            "createTime": "2025-01-01T00:00:00Z",
            "startTime": "2025-01-01T00:00:00Z",
            "updateTime": "2025-01-01T00:00:00Z"
        })))
    } else {
        if params.job_id.contains("error") {
            return Ok(Json(serde_json::json!({
                "name": format!("projects/{}/locations/{}/tuningJobs/{}", params.project_id, params.region, params.job_id),
                "state": "JOB_STATE_FAILED",
                "createTime": "2025-01-01T00:00:00Z",
                "startTime": "2025-01-01T00:00:00Z",
                "updateTime": "2025-01-01T00:00:00Z",
                "error": {
                    "message": "Model error"
                }
            })));
        }
        Ok(Json(serde_json::json!({
            "name": format!("projects/{}/locations/{}/tuningJobs/{}", params.project_id, params.region, params.job_id),
            "state": "JOB_STATE_SUCCEEDED",
            "createTime": "2025-01-01T00:00:00Z",
            "startTime": "2025-01-01T00:00:00Z",
            "updateTime": "2025-01-01T00:00:00Z",
            "tunedModel": {
                "endpoint": format!("projects/{}/locations/{}/endpoints/mock-tuned-model", params.project_id, params.region),
                "baseModel": job.model,
                "tunedModelDisplayName": "mock-tuned-model"
            }
        })))
    }
}