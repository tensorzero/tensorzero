use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use axum::{Json, extract::Path, http::StatusCode};
use serde::Deserialize;
use serde_json::json;

use crate::{apply_delay, error::Error};

// Storage for tuning jobs (job_name -> TuningJob)
static GCP_TUNING_JOBS: OnceLock<Mutex<HashMap<String, TuningJob>>> = OnceLock::new();

const POLLS_UNTIL_READY: usize = 2;

struct TuningJob {
    name: String,
    base_model: String,
    create_time: String,
    poll_count: usize,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct CreateTuningRequest {
    base_model: String,
    #[expect(dead_code)]
    supervised_tuning_spec: SupervisedTuningSpec,
    #[expect(dead_code)]
    tuned_model_display_name: Option<String>,
    #[expect(dead_code)]
    service_account: Option<String>,
    #[expect(dead_code)]
    encryption_spec: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SupervisedTuningSpec {
    #[expect(dead_code)]
    training_dataset_uri: String,
    #[expect(dead_code)]
    validation_dataset_uri: Option<String>,
    #[expect(dead_code)]
    hyper_parameters: Option<serde_json::Value>,
    #[expect(dead_code)]
    #[serde(rename = "export_last_checkpoint_only")]
    export_last_checkpoint_only: Option<bool>,
}

#[derive(Deserialize)]
pub struct TuningJobPathParams {
    project: String,
    location: String,
}

#[derive(Deserialize)]
pub struct TuningJobGetParams {
    project: String,
    location: String,
    job_id: String,
}

/// POST /v1/projects/{project}/locations/{location}/tuningJobs
pub async fn create_tuning_job(
    Path(params): Path<TuningJobPathParams>,
    Json(request): Json<CreateTuningRequest>,
) -> Result<Json<serde_json::Value>, Error> {
    apply_delay().await;

    let job_id = format!("mock-gcp-tuning-{}", uuid::Uuid::now_v7());
    let now = chrono::Utc::now();

    let job_name = format!(
        "projects/{}/locations/{}/tuningJobs/{}",
        params.project, params.location, job_id
    );

    let job = TuningJob {
        name: job_name.clone(),
        base_model: request.base_model.clone(),
        create_time: now.to_rfc3339(),
        poll_count: 0,
    };

    let mut jobs = GCP_TUNING_JOBS
        .get_or_init(Default::default)
        .lock()
        .unwrap();
    jobs.insert(job_name.clone(), job);

    Ok(Json(json!({
        "name": job_name,
        "state": "JOB_STATE_QUEUED",
        "createTime": now.to_rfc3339(),
        "baseModel": request.base_model,
    })))
}

/// GET /v1/projects/{project}/locations/{location}/tuningJobs/{job_id}
pub async fn get_tuning_job(
    Path(params): Path<TuningJobGetParams>,
) -> Result<Json<serde_json::Value>, Error> {
    apply_delay().await;

    let job_name = format!(
        "projects/{}/locations/{}/tuningJobs/{}",
        params.project, params.location, params.job_id
    );

    let mut jobs = GCP_TUNING_JOBS
        .get_or_init(Default::default)
        .lock()
        .unwrap();

    let job = jobs.get_mut(&job_name).ok_or_else(|| {
        Error::new(
            format!("Tuning job not found: {job_name}"),
            StatusCode::NOT_FOUND,
        )
    })?;

    job.poll_count += 1;

    // Check if the model name contains "error" to simulate failures
    if job.base_model.contains("error") {
        return Ok(Json(json!({
            "name": job.name,
            "state": "JOB_STATE_FAILED",
            "createTime": job.create_time,
            "baseModel": job.base_model,
            "error": {
                "code": 400,
                "message": "Tuning job failed because the model name contains 'error'"
            }
        })));
    }

    // Progress through states based on poll count
    if job.poll_count >= POLLS_UNTIL_READY {
        let endpoint_id = format!("mock-endpoint-{}", uuid::Uuid::now_v7());
        Ok(Json(json!({
            "name": job.name,
            "state": "JOB_STATE_SUCCEEDED",
            "createTime": job.create_time,
            "baseModel": job.base_model,
            "tunedModel": {
                "model": format!("projects/{}/locations/{}/models/mock-tuned-model", params.project, params.location),
                "endpoint": format!("projects/{}/locations/{}/endpoints/{}", params.project, params.location, endpoint_id)
            },
            "tuning_data_statistics": {
                "supervisedTuningDataStats": {
                    "tuningDatasetExampleCount": 100,
                    "totalBillableTokenCount": 50000,
                    "tuningStepCount": 50
                }
            }
        })))
    } else {
        Ok(Json(json!({
            "name": job.name,
            "state": "JOB_STATE_RUNNING",
            "createTime": job.create_time,
            "baseModel": job.base_model,
        })))
    }
}
