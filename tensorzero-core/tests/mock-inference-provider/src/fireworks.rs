use std::{
    collections::{hash_map::Entry, HashMap},
    sync::{Mutex, OnceLock},
};

use anyhow::anyhow;
use axum::http::StatusCode;
use axum::{
    extract::{Multipart, Path},
    response::IntoResponse,
    Json,
};
use rand::distr::{Alphanumeric, SampleString};
use serde::Deserialize;

use crate::apply_delay;

pub struct FireworksError(anyhow::Error);

impl<E> From<E> for FireworksError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

impl IntoResponse for FireworksError {
    fn into_response(self) -> axum::response::Response {
        let status = axum::http::StatusCode::INTERNAL_SERVER_ERROR;
        let body = format!("Internal Server Error: {}", self.0);
        (status, body).into_response()
    }
}

type Result<T> = std::result::Result<T, FireworksError>;

static FIREWORKS_DATASETS: OnceLock<Mutex<HashMap<DatasetKey, Dataset>>> = OnceLock::new();

static FIREWORKS_FINE_TUNING_JOBS: OnceLock<Mutex<HashMap<FineTuningJobKey, FineTuningJob>>> =
    OnceLock::new();

static FIREWORKS_DEPLOYED_MODELS: OnceLock<Mutex<HashMap<DeployedModelKey, DeployedModel>>> =
    OnceLock::new();

#[derive(Hash, PartialEq, Eq)]
struct DatasetKey {
    account_id: String,
    dataset_id: String,
}

#[derive(Deserialize)]
pub struct AccountIdParams {
    account_id: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateDatasetBody {
    dataset_id: String,
}

pub struct Dataset {
    pub has_file: bool,
    pub poll_count: usize,
}

pub async fn create_dataset(
    Path(params): Path<AccountIdParams>,
    Json(body): Json<CreateDatasetBody>,
) -> Json<serde_json::Value> {
    apply_delay().await;
    let key = DatasetKey {
        account_id: params.account_id.clone(),
        dataset_id: body.dataset_id,
    };
    FIREWORKS_DATASETS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .insert(
            key,
            Dataset {
                has_file: false,
                poll_count: 0,
            },
        );
    Json(serde_json::json!({}))
}

#[derive(Deserialize)]
pub struct DatasetKeyParams {
    account_id: String,
    dataset_id: String,
}

const POLLS_UNTIL_READY: usize = 2;

pub async fn get_dataset(Path(params): Path<DatasetKeyParams>) -> Result<Json<serde_json::Value>> {
    apply_delay().await;
    let mut datasets = FIREWORKS_DATASETS
        .get_or_init(Default::default)
        .lock()
        .unwrap();
    let dataset = datasets
        .get_mut(&DatasetKey {
            account_id: params.account_id,
            dataset_id: params.dataset_id,
        })
        .ok_or_else(|| anyhow!("Dataset not found"))?;

    dataset.poll_count += 1;
    if dataset.poll_count < POLLS_UNTIL_READY {
        Ok(Json(serde_json::json!({
            "state": "PENDING",
            "poll_count": dataset.poll_count,
        })))
    } else {
        Ok(Json(serde_json::json!({
            "state": "READY",
            "poll_count": dataset.poll_count,
        })))
    }
}

#[axum::debug_handler]
pub async fn upload_to_dataset(
    Path(params): Path<DatasetKeyParams>,
    mut form: Multipart,
) -> Result<Json<serde_json::Value>> {
    apply_delay().await;
    // We can't specify the :upload suffix in the axum route, so strip it here
    let dataset_id = params
        .dataset_id
        .strip_suffix(":upload")
        .ok_or_else(|| anyhow!("Dataset ID must end with :upload"))?;
    let field = form
        .next_field()
        .await?
        .ok_or_else(|| anyhow!("Upload endpoint requires a multipart field"))?;

    if field.content_type() != Some("application/jsonl") {
        return Err(anyhow!(
            "Expected content_type application/jsonl but found {:?}",
            field.content_type()
        )
        .into());
    }

    let _contents = field.bytes().await?;

    let next_field = form.next_field().await?;
    if next_field.is_some() {
        return Err(anyhow!("Expected only one field in multipart form").into());
    }

    FIREWORKS_DATASETS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .get_mut(&DatasetKey {
            account_id: params.account_id.clone(),
            dataset_id: dataset_id.to_string(),
        })
        .ok_or_else(|| anyhow!("Dataset not found"))?
        .has_file = true;

    Ok(Json(serde_json::json!({"filenameToSignedUrls": {}})))
}

struct FineTuningJob {
    poll_count: usize,
}

#[derive(Hash, Eq, PartialEq)]
struct FineTuningJobKey {
    account_id: String,
    job_id: String,
}

pub async fn create_fine_tuning_job(
    Path(params): Path<AccountIdParams>,
    _body: Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>> {
    apply_delay().await;
    let account_id = params.account_id.clone();
    let job_id = format!(
        "mock-fireworks-{}",
        Alphanumeric.sample_string(&mut rand::rng(), 10)
    );
    let path = format!("accounts/{account_id}/supervisedFineTuningJobs/{job_id}");

    FIREWORKS_FINE_TUNING_JOBS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .insert(
            FineTuningJobKey {
                account_id,
                job_id: job_id.clone(),
            },
            FineTuningJob { poll_count: 0 },
        );

    Ok(Json(serde_json::json!({
        "name": path,
        "state": "JOB_STATE_UNSPECIFIED",
        "status": null,
        "outputModel": null,
    })))
}

#[derive(Deserialize)]
pub struct CreatedDeployedModelParams {
    model: String,
}

#[derive(Hash, Eq, PartialEq)]
struct DeployedModelKey {
    account_id: String,
    model_id: String,
}

struct DeployedModel {
    poll_count: usize,
}

pub async fn create_deployed_model(
    Path(params): Path<AccountIdParams>,
    Json(body): Json<CreatedDeployedModelParams>,
) -> Result<(StatusCode, Json<serde_json::Value>)> {
    apply_delay().await;
    let account_id = params.account_id.clone();
    let model_id = body.model;

    let mut models = FIREWORKS_DEPLOYED_MODELS
        .get_or_init(Default::default)
        .lock()
        .unwrap();

    let entry = models.entry(DeployedModelKey {
        account_id: account_id.clone(),
        model_id: model_id.clone(),
    });

    let model_exists = matches!(entry, Entry::Occupied(_));
    // This endpoint is used for both creating and polling model deployments.
    // The status code is used to indicate whether or not the model exists.
    let status_code = if model_exists {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    };
    let model = entry
        .and_modify(|m| m.poll_count += 1)
        .or_insert(DeployedModel { poll_count: 0 });

    if model.poll_count < POLLS_UNTIL_READY {
        Ok((
            status_code,
            Json(serde_json::json!({
                "state": "DEPLOYING"
            })),
        ))
    } else {
        Ok((
            status_code,
            Json(serde_json::json!({
                "state": "DEPLOYED",
            })),
        ))
    }
}

#[derive(Deserialize)]
#[expect(dead_code)]
pub struct DeployedModelParams {
    account_id: String,
    model_id: String,
}

#[derive(Deserialize)]
pub struct JobPathParams {
    account_id: String,
    job_id: String,
}

pub async fn get_fine_tuning_job(
    Path(params): Path<JobPathParams>,
) -> Result<Json<serde_json::Value>> {
    apply_delay().await;
    let mut jobs = FIREWORKS_FINE_TUNING_JOBS
        .get_or_init(Default::default)
        .lock()
        .unwrap();
    let job = jobs
        .get_mut(&FineTuningJobKey {
            account_id: params.account_id.clone(),
            job_id: params.job_id.clone(),
        })
        .ok_or_else(|| anyhow!("Job not found"))?;

    job.poll_count += 1;
    if job.poll_count < POLLS_UNTIL_READY {
        Ok(Json(serde_json::json!({
            "name": format!("accounts/{}/supervisedFineTuningJobs/{}", params.account_id, params.job_id),
            "state": "JOB_STATE_RUNNING",
        })))
    } else {
        if params.account_id.contains("error") {
            return Ok(Json(serde_json::json!({
                "name": format!("accounts/{}/supervisedFineTuningJobs/{}", params.account_id, params.job_id),
                "state": "JOB_STATE_FAILED",
                "error": serde_json::json!({
                    "unexpected_error": "Model error"
                }),
            })));
        }
        Ok(Json(serde_json::json!({
            "name": format!("accounts/{}/supervisedFineTuningJobs/{}", params.account_id, params.job_id),
            "state": "JOB_STATE_COMPLETED",
            "outputModel": format!("accounts/{}/models/mock-fireworks-model", params.account_id),
        })))
    }
}
