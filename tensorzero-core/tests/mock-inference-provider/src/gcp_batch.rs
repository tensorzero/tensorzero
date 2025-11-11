use axum::{extract::Path, http::StatusCode, Json};
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
use uuid::Uuid;

use crate::{apply_delay, error::Error};

// Storage for batch jobs (job_id -> BatchJob)
static GCP_BATCH_JOBS: OnceLock<Mutex<HashMap<String, BatchJob>>> = OnceLock::new();

#[derive(Clone)]
struct BatchJob {
    name: String, // Full resource name: projects/{project}/locations/{location}/batchPredictionJobs/{id}
    display_name: String,
    model: String,
    state: JobState,
    input_config: InputConfig,
    output_config: OutputConfig,
    create_time: String,
    // Timestamp when this job should transition to succeeded
    complete_at: i64,
}

#[derive(Clone, Debug)]
enum JobState {
    Pending,
    Running,
    Succeeded,
}

impl JobState {
    fn as_str(&self) -> &'static str {
        match self {
            JobState::Pending => "JOB_STATE_PENDING",
            JobState::Running => "JOB_STATE_RUNNING",
            JobState::Succeeded => "JOB_STATE_SUCCEEDED",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InputConfig {
    #[serde(rename = "instancesFormat")]
    instances_format: String,
    gcs_source: GCSSource,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GCSSource {
    uris: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OutputConfig {
    #[serde(rename = "predictionsFormat")]
    predictions_format: String,
    gcs_destination: GCSDestination,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GCSDestination {
    output_uri_prefix: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateBatchRequest {
    display_name: String,
    model: String,
    input_config: InputConfig,
    output_config: OutputConfig,
}

/// POST /v1/projects/{project}/locations/{location}/batchPredictionJobs
pub async fn create_batch_prediction_job(
    Path((project, location)): Path<(String, String)>,
    Json(request): Json<CreateBatchRequest>,
) -> Result<Json<serde_json::Value>, Error> {
    apply_delay().await;

    let job_id = Uuid::now_v7();
    let now = chrono::Utc::now();
    let complete_at = now.timestamp() + 2; // Complete after 2 seconds

    let job_name = format!(
        "projects/{}/locations/{}/batchPredictionJobs/{}",
        project, location, job_id
    );

    let batch = BatchJob {
        name: job_name.clone(),
        display_name: request.display_name.clone(),
        model: request.model.clone(),
        state: JobState::Pending,
        input_config: request.input_config,
        output_config: request.output_config,
        create_time: now.to_rfc3339(),
        complete_at,
    };

    let mut jobs = GCP_BATCH_JOBS.get_or_init(Default::default).lock().unwrap();
    jobs.insert(job_name.clone(), batch);
    drop(jobs);

    Ok(Json(json!({
        "name": job_name,
        "displayName": request.display_name,
        "model": request.model,
        "state": "JOB_STATE_PENDING",
        "createTime": now.to_rfc3339(),
    })))
}

/// GET /v1/projects/{project}/locations/{location}/batchPredictionJobs/{job_id}
pub async fn get_batch_prediction_job(
    Path((project, location, job_id)): Path<(String, String, String)>,
) -> Result<Json<serde_json::Value>, Error> {
    apply_delay().await;

    let job_name = format!(
        "projects/{}/locations/{}/batchPredictionJobs/{}",
        project, location, job_id
    );

    let mut jobs = GCP_BATCH_JOBS.get_or_init(Default::default).lock().unwrap();
    let job = jobs.get_mut(&job_name).ok_or_else(|| {
        Error::new(
            format!("Batch prediction job not found: {}", job_name),
            StatusCode::NOT_FOUND,
        )
    })?;

    let now = chrono::Utc::now().timestamp();

    // Update job state based on time
    if now >= job.complete_at && !matches!(job.state, JobState::Succeeded) {
        // Transition to succeeded
        job.state = JobState::Succeeded;
    } else if now >= job.complete_at - 1 && matches!(job.state, JobState::Pending) {
        // After 1 second, transition to running
        job.state = JobState::Running;
    }

    let mut response = json!({
        "name": job.name,
        "displayName": job.display_name,
        "model": job.model,
        "state": job.state.as_str(),
        "createTime": job.create_time,
    });

    // Add output info if job is succeeded
    if matches!(job.state, JobState::Succeeded) {
        response["outputInfo"] = json!({
            "gcsOutputDirectory": job.output_config.gcs_destination.output_uri_prefix
        });
    }

    Ok(Json(response))
}
