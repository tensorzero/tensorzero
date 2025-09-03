use axum::{
    extract::{Json, Path},
    response::Json as ResponseJson,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
use uuid::{NoContext, Timestamp, Uuid};

#[derive(Clone)]
pub struct GCPFineTuningJob {
    pub num_polls: usize,
    pub val: serde_json::Value,
    pub finish_at: Option<chrono::DateTime<chrono::Utc>>,
}

static GCP_FINE_TUNING_JOBS: OnceLock<Mutex<HashMap<String, GCPFineTuningJob>>> = OnceLock::new();

#[derive(Deserialize)]
pub struct CreateJobPath {
    project_id: String,
    region: String,
}

#[derive(Deserialize)]
pub struct GetJobPath {
    project_id: String,
    region: String,
    job_name: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GCPFineTuningRequest {
    #[expect(dead_code)]
    base_model: String,
    supervised_tuning_spec: SupervisedTuningSpec,
    tuned_model_display_name: Option<String>,
    encryption_spec: Option<EncryptionSpec>,
    service_account: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct SupervisedTuningSpec {
    training_dataset_uri: String,
    validation_dataset_uri: Option<String>,
    hyper_parameters: Option<HyperParameters>,
    #[serde(rename = "export_last_checkpoint_only")]
    export_last_checkpoint_only: Option<bool>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct HyperParameters {
    epoch_count: Option<u64>,
    adapter_size: Option<u64>, // Changed from String to u64 to match the actual type
    learning_rate_multiplier: Option<f64>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct EncryptionSpec {
    kms_key_name: Option<String>,
}

const SHOW_PROGRESS_AT: usize = 1;

pub async fn create_fine_tuning_job(
    Path(params): Path<CreateJobPath>,
    Json(payload): Json<GCPFineTuningRequest>,
) -> ResponseJson<Value> {
    let job_id = format!(
        "projects/{}/locations/{}/tuningJobs/{}",
        params.project_id,
        params.region,
        Uuid::new_v7(Timestamp::now(NoContext))
    );

    let response = json!({
        "name": job_id,
        "createTime": chrono::Utc::now().to_rfc3339(),
        "updateTime": chrono::Utc::now().to_rfc3339(),
        "state": "JOB_STATE_QUEUED",
        "supervisedTuningSpec": payload.supervised_tuning_spec,
        "tunedModelDisplayName": payload.tuned_model_display_name,
        "encryptionSpec": payload.encryption_spec,
        "serviceAccount": payload.service_account,
    });

    let job = GCPFineTuningJob {
        num_polls: 0,
        val: response.clone(),
        finish_at: None,
    };

    let mut fine_tuning_jobs = GCP_FINE_TUNING_JOBS
        .get_or_init(Default::default)
        .lock()
        .unwrap();
    fine_tuning_jobs.insert(job_id.clone(), job);

    ResponseJson(response)
}

pub async fn get_fine_tuning_job(Path(params): Path<GetJobPath>) -> ResponseJson<Value> {
    // Extract the actual job ID from the full job name path
    let job_name = params.job_name.trim_start_matches('/');
    let full_job_name = format!(
        "projects/{}/locations/{}/tuningJobs/{}",
        params.project_id, params.region, job_name
    );

    let mut fine_tuning_jobs = GCP_FINE_TUNING_JOBS
        .get_or_init(Default::default)
        .lock()
        .unwrap();

    let job = fine_tuning_jobs.get_mut(&full_job_name);
    if let Some(job) = job {
        job.num_polls += 1;

        if job.num_polls == SHOW_PROGRESS_AT {
            let finish_at = chrono::Utc::now() + chrono::Duration::seconds(2);
            job.finish_at = Some(finish_at);
            job.val["estimatedFinish"] = finish_at.timestamp().into();
        }

        if let Some(finish_at) = job.finish_at {
            if chrono::Utc::now() >= finish_at {
                job.val["state"] = "JOB_STATE_SUCCEEDED".into();
                job.val["endTime"] = chrono::Utc::now().to_rfc3339().into();
                job.val["tunedModel"] = json!({
                    "model": format!("projects/{}/locations/{}/models/{}",
                        params.project_id, params.region,
                        Uuid::new_v7(Timestamp::now(NoContext))),
                    "endpoint": format!("projects/{}/locations/{}/endpoints/{}",
                        params.project_id,
                        params.region,
                        Uuid::new_v7(Timestamp::now(NoContext)))
                });
                job.val["experiment"] = format!(
                    "projects/{}/locations/{}/metadataStores/default/contexts/{}",
                    params.project_id,
                    params.region,
                    Uuid::new_v7(Timestamp::now(NoContext))
                )
                .into();
                job.val["tuning_data_statistics"] = json!({
                    "supervisedTuningDataStats": {
                        "tuningDatasetExampleCount": 1000,
                        "totalBillableTokenCount": 50000,
                        "tuningStepCount": 100
                    }
                });
            } else {
                job.val["state"] = "JOB_STATE_RUNNING".into();
            }
        } else {
            job.val["state"] = "JOB_STATE_PENDING".into();
        }

        ResponseJson(job.val.clone())
    } else {
        ResponseJson(json!({
            "error": {
                "code": 404,
                "message": format!("Job {} not found", full_job_name),
                "state": "NOT_FOUND"
            }
        }))
    }
}
