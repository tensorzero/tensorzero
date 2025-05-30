use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::http;
use futures::StreamExt;
use google_cloud_auth::credentials::Credentials;
use http::{HeaderMap, HeaderValue};
use itertools::Itertools;
use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use object_store::gcp::{GcpCredential, GoogleCloudStorageBuilder};
use object_store::{ObjectStore, StaticCredentialProvider};
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use serde_json::Value;
use tokio::time::Instant;
use url::Url;
use uuid::Uuid;

use crate::cache::ModelProviderRequest;
use crate::config_parser::{
    GCPBatchConfigCloudStorage, GCPBatchConfigType, GCPProviderTypeConfig, ProviderTypesConfig,
};
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{DisplayOrDebugGateway, Error, ErrorDetails};
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::batch::{
    BatchRequestRow, BatchStatus, PollBatchInferenceResponse, ProviderBatchInferenceOutput,
    ProviderBatchInferenceResponse,
};
use crate::inference::types::resolved_input::FileWithPath;
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, serialize_or_log, ModelInferenceRequest,
    PeekableProviderInferenceResponseStream, ProviderInferenceResponse,
    ProviderInferenceResponseChunk, RequestMessage, Usage,
};
use crate::inference::types::{
    ContentBlock, ContentBlockChunk, ContentBlockOutput, FileKind, FinishReason, FlattenUnknown,
    Latency, ModelInferenceRequestJsonMode, ProviderInferenceResponseArgs,
    ProviderInferenceResponseStreamInner, Role, Text, TextChunk,
};
use crate::model::{
    build_creds_caching_default_with_fn, fully_qualified_name, Credential, CredentialLocation,
    ModelProvider,
};
use crate::tool::{ToolCall, ToolCallChunk, ToolChoice, ToolConfig};

use super::gcp_vertex_anthropic::make_gcp_sdk_credentials;
use super::helpers::{inject_extra_request_data, parse_jsonl_batch_file, JsonlBatchFileInfo};
use super::openai::convert_stream_error;

const PROVIDER_NAME: &str = "GCP Vertex Gemini";
const PROVIDER_TYPE: &str = "gcp_vertex_gemini";

const INFERENCE_ID_LABEL: &str = "tensorzero::inference_id";

/// Implements a subset of the GCP Vertex Gemini API as documented [here](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.publishers.models/generateContent) for non-streaming
/// and [here](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.publishers.models/streamGenerateContent) for streaming
#[derive(Debug)]
pub struct GCPVertexGeminiProvider {
    api_v1_base_url: Url,
    request_url: String,
    streaming_request_url: String,
    audience: String,
    credentials: GCPVertexCredentials,
    model_id: String,
    batch_config: Option<BatchConfig>,
}

#[derive(Debug)]
struct BatchConfig {
    input_uri_prefix: String,
    output_uri_prefix: String,
    batch_request_url: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiBatchRequest {
    display_name: String,
    model: String,
    input_config: GCPVertexGeminiBatchRequestInputConfig,
    output_config: GCPVertexGeminiBatchRequestOutputConfig,
}

#[derive(Serialize)]
#[serde(tag = "predictionsFormat", rename_all = "camelCase")]
enum GCPVertexGeminiBatchRequestOutputConfig {
    #[serde(rename = "jsonl")]
    Jsonl {
        gcs_destination: GCPVertexGeminiGCSDestination,
    },
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiGCSDestination {
    output_uri_prefix: String,
}

#[derive(Serialize)]
#[serde(tag = "instancesFormat", rename_all = "camelCase")]
enum GCPVertexGeminiBatchRequestInputConfig {
    #[serde(rename = "jsonl")]
    Jsonl {
        gcs_source: GCPVertexGeminiGCSSource,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiGCSSource {
    uris: String,
}

static DEFAULT_CREDENTIALS: OnceLock<GCPVertexCredentials> = OnceLock::new();

struct StoreAndPath {
    store: Box<dyn ObjectStore>,
    path: object_store::path::Path,
}

/// Joins a Google Cloud Storage directory path (with an optional trailing slash) with a file name.
/// This is used to support both "gs://bucket/path" and "gs://bucket/path/" formats in tensorzero.toml
fn join_cloud_paths(dir: &str, file: &str) -> String {
    dir.strip_suffix("/").unwrap_or(dir).to_string() + "/" + file
}

/// Constructs a new `ObjectStore` instance for the bucket specified by `gs_url`
/// with the provided credentials.
/// We call this on each batch start/poll request, as we might be using dynamic credentials.
async fn make_gcp_object_store(
    gs_url: &str,
    credentials: &GCPVertexCredentials,
    dynamic_api_keys: &InferenceCredentials,
) -> Result<StoreAndPath, Error> {
    let bucket_and_path = gs_url.strip_prefix("gs://").ok_or_else(|| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Google Cloud Storage url does not start with 'gs://': {gs_url}"),
        })
    })?;
    let (bucket, path) = bucket_and_path.split_once("/").ok_or_else(|| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Google Cloud Storage url does not contain a bucket name: {gs_url}"),
        })
    })?;
    let key = object_store::path::Path::parse(path).map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to parse Google Cloud Storage path: {e}"),
        })
    })?;

    let mut builder = GoogleCloudStorageBuilder::default().with_bucket_name(bucket);

    match credentials {
        GCPVertexCredentials::Static { raw, parsed: _ } => {
            builder = builder.with_service_account_key(raw.expose_secret())
        }
        GCPVertexCredentials::Dynamic(key_name) => {
            let key = dynamic_api_keys.get(key_name).ok_or_else(|| {
                Error::new(ErrorDetails::ApiKeyMissing {
                    provider_name: PROVIDER_NAME.to_string(),
                })
            })?;
            builder =
                builder.with_credentials(Arc::new(StaticCredentialProvider::new(GcpCredential {
                    bearer: key.expose_secret().to_string(),
                })));
        }
        GCPVertexCredentials::Sdk(creds) => {
            let headers = creds
                .headers(http::Extensions::default())
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::GCPCredentials {
                        message: format!("Failed to get GCP access token: {e}"),
                    })
                })?;

            // The 'object_store' crate requires us to use a bearer auth token, so try to extract that from the produced headers
            // In the future, we may want to use the GCP object store sdk crate directly, so that we can support all of the
            // auth methods
            if headers.len() != 1 {
                return Err(Error::new(ErrorDetails::GCPCredentials {
                    message: format!(
                        "Expected GCP SDK to produce exactly one auth headers, found: {:?}",
                        headers.keys()
                    ),
                }));
            }

            let header_value = headers.get("Authorization").ok_or_else(|| {
                Error::new(ErrorDetails::GCPCredentials {
                    message: format!(
                        "Expected GCP SDK to produce an Authorization header, found: {:?}",
                        headers.keys()
                    ),
                })
            })?;

            if let Some(bearer_token) = header_value
                .to_str()
                .ok()
                .and_then(|s| s.strip_prefix("Bearer "))
            {
                builder = builder.with_credentials(Arc::new(StaticCredentialProvider::new(
                    GcpCredential {
                        bearer: bearer_token.to_string(),
                    },
                )));
            } else {
                return Err(Error::new(ErrorDetails::GCPCredentials {
                    message:
                        "Expected GCP SDK to produce a Bearer token in the Authorization header"
                            .to_string(),
                }));
            }
        }
        GCPVertexCredentials::None => {
            return Err(Error::new(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
            }))
        }
    }

    let store = builder.build().map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to create GCS object store: {e}"),
        })
    })?;

    Ok(StoreAndPath {
        store: Box::new(store),
        path: key,
    })
}

#[derive(Debug, Eq, PartialEq)]
pub enum ShorthandUrl<'a> {
    // We enforce that the publisher is 'google' or 'anthropic' when parsing the url,
    // depending on which model provider is parsing the shorthand url.
    Publisher {
        location: &'a str,
        model_id: &'a str,
    },
    Endpoint {
        location: &'a str,
        endpoint_id: &'a str,
    },
}

// Parses strings of the form:
// * 'projects/<project_id>/locations/<location>/publishers/<publisher>/models/<model_id>'
// * 'projects/<project_id>/locations/<location>/endpoints/<endpoint_id>'
pub fn parse_shorthand_url<'a>(
    shorthand_url: &'a str,
    expected_publisher: &str,
) -> Result<ShorthandUrl<'a>, Error> {
    let components: Vec<&str> = shorthand_url.split('/').collect_vec();
    let [projects, _project_id, locations, location, publishers_or_endpoint, ..] = &components[..]
    else {
        return Err(Error::new(ErrorDetails::Config {
            message: format!("GCP shorthand url is not in the expected format (should start with `projects/<project_id>/locations/<location>'): `{shorthand_url}`"),
        }));
    };

    if projects != &"projects" {
        return Err(Error::new(ErrorDetails::Config {
            message: format!("GCP shorthand url does not start with 'projects': `{shorthand_url}`"),
        }));
    }
    if locations != &"locations" {
        return Err(Error::new(ErrorDetails::Config {
            message: format!("GCP shorthand url does contain '/locations/': `{shorthand_url}`"),
        }));
    }

    if publishers_or_endpoint == &"publishers" {
        let publisher = components.get(5).ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "GCP shorthand url does not contain a publisher: `{shorthand_url}`"
                ),
            })
        })?;
        if *publisher != expected_publisher {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "GCP shorthand url has publisher `{publisher}`, expected `{expected_publisher}` : `{shorthand_url}`"
                ),
            }));
        }
        let models = components.get(6).ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "GCP shorthand url does not contain a model or endpoint: `{shorthand_url}`"
                ),
            })
        })?;
        if models != &"models" {
            return Err(Error::new(ErrorDetails::Config {
                message: format!("GCP shorthand url does not contain a model: `{shorthand_url}`"),
            }));
        }
        let model_id = components.get(7).ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "GCP shorthand url does not contain a model id: `{shorthand_url}`"
                ),
            })
        })?;
        Ok(ShorthandUrl::Publisher { location, model_id })
    } else if publishers_or_endpoint == &"endpoints" {
        let endpoint = components.get(5).ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "GCP shorthand url does not contain an endpoint: `{shorthand_url}`"
                ),
            })
        })?;
        Ok(ShorthandUrl::Endpoint {
            endpoint_id: endpoint,
            location,
        })
    } else {
        Err(Error::new(ErrorDetails::Config {
            message: format!(
                "GCP shorthand url does not contain a publisher or endpoint: `{shorthand_url}`"
            ),
        }))
    }
}

impl GCPVertexGeminiProvider {
    // Constructs a provider from a shorthand string of the form:
    // * 'projects/<project_id>/locations/<location>/publishers/google/models/XXX'
    // * 'projects/<project_id>/locations/<location>/endpoints/XXX'
    //
    // This is *not* a full url - we append ':generateContent' or ':streamGenerateContent' to the end of the path as needed.
    pub async fn new_shorthand(project_url_path: String) -> Result<Self, Error> {
        let cred_location = default_api_key_location();
        let credentials = if matches!(cred_location, CredentialLocation::Sdk) {
            make_gcp_sdk_credentials(PROVIDER_TYPE).await?
        } else {
            build_creds_caching_default_with_fn(
                None,
                cred_location,
                PROVIDER_TYPE,
                &DEFAULT_CREDENTIALS,
                |creds| GCPVertexCredentials::try_from((creds, PROVIDER_TYPE)),
            )?
        };

        // We only support model urls with the publisher 'google' (which includes all of the Gemini models)
        let shorthand_url = parse_shorthand_url(&project_url_path, "google")?;
        let (location, model_id) = match shorthand_url {
            ShorthandUrl::Publisher { location, model_id } => (location, model_id.to_string()),
            ShorthandUrl::Endpoint {
                location,
                endpoint_id,
            } => (location, format!("endpoints/{endpoint_id}")),
        };

        let request_url = format!(
            "https://{location}-aiplatform.googleapis.com/v1/{project_url_path}:generateContent"
        );
        let streaming_request_url = format!("https://{location}-aiplatform.googleapis.com/v1/{project_url_path}:streamGenerateContent?alt=sse");
        let audience = format!("https://{location}-aiplatform.googleapis.com/");
        let api_v1_base_url = Url::parse(&format!(
            "https://{location}-aiplatform.googleapis.com/v1/"
        ))
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to parse base URL - this should never happen: {e}"),
            })
        })?;

        Ok(GCPVertexGeminiProvider {
            api_v1_base_url,
            request_url,
            streaming_request_url,
            batch_config: None,
            audience,
            credentials,
            model_id,
        })
    }

    pub async fn new(
        model_id: String,
        location: String,
        project_id: String,
        api_key_location: Option<CredentialLocation>,
        provider_types: &ProviderTypesConfig,
    ) -> Result<Self, Error> {
        let default_location = default_api_key_location();
        let cred_location = api_key_location.as_ref().unwrap_or(&default_location);

        let credentials = if matches!(cred_location, CredentialLocation::Sdk) {
            make_gcp_sdk_credentials(PROVIDER_TYPE).await?
        } else {
            build_creds_caching_default_with_fn(
                api_key_location,
                default_api_key_location(),
                PROVIDER_TYPE,
                &DEFAULT_CREDENTIALS,
                |creds| GCPVertexCredentials::try_from((creds, PROVIDER_TYPE)),
            )?
        };

        let request_url = format!("https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent");
        let streaming_request_url = format!("https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:streamGenerateContent?alt=sse");
        let audience = format!("https://{location}-aiplatform.googleapis.com/");
        let api_v1_base_url = Url::parse(&format!(
            "https://{location}-aiplatform.googleapis.com/v1/"
        ))
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to parse base URL - this should never happen: {e}"),
            })
        })?;

        let batch_config = match &provider_types.gcp_vertex_gemini {
            Some(GCPProviderTypeConfig { batch: Some(GCPBatchConfigType::CloudStorage(GCPBatchConfigCloudStorage {
                input_uri_prefix,
                output_uri_prefix,
            }))}) => {
                Some(BatchConfig {
                    input_uri_prefix: input_uri_prefix.clone(),
                    output_uri_prefix: output_uri_prefix.clone(),
                    batch_request_url: format!("https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/batchPredictionJobs"),
                })
            }
            _ => None,
        };
        Ok(GCPVertexGeminiProvider {
            api_v1_base_url,
            request_url,
            streaming_request_url,
            batch_config,
            audience,
            credentials,
            model_id,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn collect_finished_batch(
        &self,
        output_data: GCPVertexBatchResponseOutputInfo,
        raw_request: String,
        raw_response: String,
        api_key: &GCPVertexCredentials,
        dynamic_api_keys: &InferenceCredentials,
        batch_request: &BatchRequestRow<'_>,
    ) -> Result<ProviderBatchInferenceResponse, Error> {
        match output_data {
            GCPVertexBatchResponseOutputInfo::Gcs {
                gcs_output_directory,
            } => {
                // The Vertex Gemini batch job always seems to write to 'predictions.jsonl' in the output directory.
                // Note that we use the path provided in the API response, which might be different from the
                // `output_uri_prefix` path in our config (if it was changed after the job was created).
                // For now, we use the same set of credentials for writing to Google Cloud Storage as we do for invoking
                // the Vertex API. In the future, we may want to allow configuring a separate set of credentials.
                let store_and_path = make_gcp_object_store(
                    &join_cloud_paths(&gcs_output_directory, "predictions.jsonl"),
                    api_key,
                    dynamic_api_keys,
                )
                .await?;
                let data = store_and_path
                    .store
                    .get(&store_and_path.path)
                    .await
                    .map_err(|e| {
                        Error::new(ErrorDetails::InternalError {
                            message: format!("Failed to get GCS object: {e}"),
                        })
                    })?
                    .bytes()
                    .await;

                parse_jsonl_batch_file::<GCPVertexBatchResponseLine, _>(
                    data,
                    JsonlBatchFileInfo {
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request,
                        raw_response,
                        file_id: store_and_path.path.to_string(),
                    },
                    |r| {
                        make_provider_batch_inference_output(
                            r,
                            &batch_request.model_name,
                            &batch_request.model_provider_name,
                        )
                    },
                )
                .await
            }
        }
    }
}

pub fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::PathFromEnv("GCP_VERTEX_CREDENTIALS_PATH".to_string())
}

#[derive(Clone, Debug)]
pub enum GCPVertexCredentials {
    Static {
        parsed: GCPServiceAccountCredentials,
        raw: SecretString,
    },
    Dynamic(String),
    Sdk(Credentials),
    None,
}

impl TryFrom<(Credential, &str)> for GCPVertexCredentials {
    type Error = Error;

    fn try_from((credentials, model): (Credential, &str)) -> Result<Self, Error> {
        match credentials {
            Credential::FileContents(file_content) => Ok(GCPVertexCredentials::Static {
                parsed: GCPServiceAccountCredentials::from_json_str(file_content.expose_secret())
                    .map_err(|e| {
                    Error::new(ErrorDetails::GCPCredentials {
                        message: format!("Failed to load GCP credentials: {e}"),
                    })
                })?,
                raw: file_content,
            }),
            Credential::Dynamic(key_name) => Ok(GCPVertexCredentials::Dynamic(key_name)),
            Credential::Missing => Ok(GCPVertexCredentials::None),
            _ => Err(Error::new(ErrorDetails::GCPCredentials {
                message: format!("Invalid credential_location for {model} provider"),
            }))?,
        }
    }
}

#[derive(Serialize)]
struct GCPVertexBatchLine<'a> {
    request: GCPVertexGeminiRequest<'a>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexBatchResponse {
    name: String,
    state: GCPVertexJobState,
    output_info: Option<GCPVertexBatchResponseOutputInfo>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum GCPVertexBatchResponseOutputInfo {
    Gcs {
        #[serde(rename = "gcsOutputDirectory")]
        gcs_output_directory: String,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiRequestMinimal {
    #[serde(default)]
    labels: HashMap<String, String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexBatchResponseLine {
    request: Box<RawValue>,
    response: GCPVertexGeminiResponse,
}
fn make_provider_batch_inference_output(
    line: GCPVertexBatchResponseLine,
    model_name: &str,
    provider_name: &str,
) -> Result<ProviderBatchInferenceOutput, Error> {
    let raw_request = line.request.to_string();
    let request = GCPVertexGeminiRequestMinimal::deserialize(&*line.request).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Error deserializing batch request: {e}"),
        })
    })?;
    let raw_response = serde_json::to_string(&line.response).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Error serializing batch response: {e}"),
        })
    })?;
    let inference_id = request.labels.get(INFERENCE_ID_LABEL).ok_or_else(|| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Missing {INFERENCE_ID_LABEL} label on GCP batch request"),
        })
    })?;

    let usage = line
        .response
        .usage_metadata
        .clone()
        .ok_or_else(|| {
            Error::new(ErrorDetails::InferenceServer {
                message: "GCP Vertex Gemini batch response has no usage metadata".to_string(),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?
        .into();

    let (output, finish_reason) = get_response_content(
        line.response,
        &raw_request,
        &raw_response,
        model_name,
        provider_name,
    )?;
    Ok(ProviderBatchInferenceOutput {
        id: Uuid::parse_str(inference_id).map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Invalid inference ID: {e}"),
            })
        })?,
        output,
        raw_response,
        usage,
        finish_reason,
    })
}

#[derive(Clone, Copy, Debug, Deserialize)]
enum GCPVertexJobState {
    #[serde(rename = "JOB_STATE_UNSPECIFIED")]
    Unspecified,
    #[serde(rename = "JOB_STATE_QUEUED")]
    Queued,
    #[serde(rename = "JOB_STATE_PENDING")]
    Pending,
    #[serde(rename = "JOB_STATE_RUNNING")]
    Running,
    #[serde(rename = "JOB_STATE_SUCCEEDED")]
    Succeeded,
    #[serde(rename = "JOB_STATE_FAILED")]
    Failed,
    #[serde(rename = "JOB_STATE_CANCELING")]
    Cancelling,
    #[serde(rename = "JOB_STATE_CANCELLED")]
    Cancelled,
    #[serde(rename = "JOB_STATE_PAUSED")]
    Paused,
    #[serde(rename = "JOB_STATE_EXPIRED")]
    Expired,
    #[serde(rename = "JOB_STATE_UPDATING")]
    Updating,
    #[serde(rename = "JOB_STATE_PARTIALLY_SUCCEEDED")]
    PartiallySucceeded,
    #[serde(other)]
    Unknown,
}

impl GCPVertexCredentials {
    pub async fn get_auth_headers<'a>(
        &'a self,
        audience: &'a str,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<HeaderMap, Error> {
        let bearer_token = match self {
            GCPVertexCredentials::Static { parsed, raw: _ } => {
                Cow::Owned(parsed.get_jwt_token(audience)?)
            }
            GCPVertexCredentials::Dynamic(key_name) => Cow::Borrowed(
                dynamic_api_keys
                    .get(key_name)
                    .ok_or_else(|| {
                        Error::new(ErrorDetails::ApiKeyMissing {
                            provider_name: PROVIDER_NAME.to_string(),
                        })
                    })?
                    .expose_secret(),
            ),
            GCPVertexCredentials::Sdk(creds) => {
                return creds
                    .headers(http::Extensions::default())
                    .await
                    .map_err(|e| {
                        Error::new(ErrorDetails::GCPCredentials {
                            message: format!("Failed to get GCP access token: {e}"),
                        })
                    })
            }
            GCPVertexCredentials::None => {
                return Err(Error::new(ErrorDetails::ApiKeyMissing {
                    provider_name: PROVIDER_NAME.to_string(),
                }))
            }
        };
        let mut headers = HeaderMap::new();
        headers.insert(
            "Authorization",
            HeaderValue::from_str(&format!("Bearer {bearer_token}",)).map_err(|e| {
                Error::new(ErrorDetails::GCPCredentials {
                    message: format!(
                        "Failed to create GCP Vertex Gemini credentials from SDK: {e}",
                    ),
                })
            })?,
        );
        Ok(headers)
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct GCPVertexBatchParams {
    job_url_suffix: String,
}
/// Auth
///
/// We implement below the JWT request signing as documented [here](https://developers.google.com/identity/protocols/oauth2/service-account).
///
/// GCPCredentials contains the pieces of information required to successfully make a request using a service account JWT
/// key. The way this works is that there are "claims" about who is making the request and we sign those claims using the key.
#[derive(Clone)]
pub struct GCPServiceAccountCredentials {
    pub private_key_id: String,
    pub private_key: EncodingKey,
    pub client_email: String,
}

impl std::fmt::Debug for GCPServiceAccountCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GCPCredentials")
            .field("private_key_id", &self.private_key_id)
            .field("private_key", &"[redacted]")
            .field("client_email", &self.client_email)
            .finish()
    }
}

/// JWT standard claims that are used in GCP auth.
#[derive(Serialize)]
struct Claims<'a> {
    iss: &'a str, // Issuer
    sub: &'a str, // Subject
    aud: &'a str, // Audience
    iat: u64,     // Issued at
    exp: u64,     // Expiration time
}

impl<'a> Claims<'a> {
    fn new(iss: &'a str, sub: &'a str, aud: &'a str) -> Self {
        #[expect(clippy::expect_used)]
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards");
        let iat = current_time.as_secs();
        let exp = (current_time + Duration::from_secs(3600)).as_secs();
        Self {
            iss,
            sub,
            aud,
            iat,
            exp,
        }
    }
}

impl GCPServiceAccountCredentials {
    // Parse a JSON string into a GCPServiceAccountCredentials struct that can be used to sign requests.
    pub fn from_json_str(credential_str: &str) -> Result<Self, Error> {
        let credential_value: Value = serde_json::from_str(credential_str).map_err(|e| {
            Error::new(ErrorDetails::GCPCredentials {
                message: format!(
                    "Failed to parse GCP Vertex Gemini credentials: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        match (
            credential_value
                .get("private_key_id")
                .ok_or_else(|| {
                    Error::new(ErrorDetails::GCPCredentials {
                        message: "GCP Vertex Gemini: missing private_key_id".to_string(),
                    })
                })?
                .as_str(),
            credential_value
                .get("private_key")
                .ok_or_else(|| {
                    Error::new(ErrorDetails::GCPCredentials {
                        message: "GCP Vertex Gemini: missing private_key".to_string(),
                    })
                })?
                .as_str(),
            credential_value
                .get("client_email")
                .ok_or_else(|| {
                    Error::new(ErrorDetails::GCPCredentials {
                        message: "GCP Vertex Gemini: missing client_email".to_string(),
                    })
                })?
                .as_str(),
        ) {
            (Some(private_key_id), Some(private_key), Some(client_email)) => {
                Ok(GCPServiceAccountCredentials {
                    private_key_id: private_key_id.to_string(),
                    private_key: EncodingKey::from_rsa_pem(private_key.as_bytes()).map_err(
                        |_| {
                            Error::new(ErrorDetails::GCPCredentials {
                                message: "GCP Vertex Gemini: private_key failed to parse as RSA"
                                    .to_string(),
                            })
                        },
                    )?,
                    client_email: client_email.to_string(),
                })
            }
            _ => Err(ErrorDetails::GCPCredentials {
                message: "GCP Vertex Gemini: missing required credentials".to_string(),
            }
            .into()),
        }
    }

    // Get a signed JWT token for the given audience valid from the current time.
    pub fn get_jwt_token(&self, audience: &str) -> Result<String, Error> {
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some(self.private_key_id.clone());
        let claims = Claims::new(&self.client_email, &self.client_email, audience);
        let token = encode(&header, &claims, &self.private_key).map_err(|e| {
            Error::new(ErrorDetails::GCPCredentials {
                message: format!("Failed to encode JWT: {}", DisplayOrDebugGateway::new(e)),
            })
        })?;
        Ok(token)
    }
}

impl InferenceProvider for GCPVertexGeminiProvider {
    /// GCP Vertex Gemini non-streaming API request
    async fn infer<'a>(
        &'a self,
        provider_request: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let mut request_body = serde_json::to_value(GCPVertexGeminiRequest::new(
            provider_request.request,
            &self.model_id,
        )?)
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing GCP Vertex Gemini request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let headers = inject_extra_request_data(
            &provider_request.request.extra_body,
            &provider_request.request.extra_headers,
            model_provider,
            provider_request.model_name,
            &mut request_body,
        )?;
        let auth_headers = self
            .credentials
            .get_auth_headers(&self.audience, dynamic_api_keys)
            .await?;
        let start_time = Instant::now();
        let res = http_client
            .post(&self.request_url)
            .json(&request_body)
            .headers(auth_headers)
            .headers(headers)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!("Error sending request: {}", DisplayOrDebugGateway::new(e)),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;

            let response = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {raw_response}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: Some(raw_response.clone()),
                })
            })?;
            let response_with_latency = GCPVertexGeminiResponseWithMetadata {
                response,
                latency,
                request: request_body,
                generic_request: provider_request.request,
                raw_response,
                model_name: provider_request.model_name,
                provider_name: provider_request.provider_name,
            };
            Ok(response_with_latency.try_into()?)
        } else {
            let response_code = res.status();
            let error_body = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
            Err(handle_gcp_vertex_gemini_error(
                serde_json::to_string(&request_body).unwrap_or_default(),
                response_code,
                error_body,
            ))
        }
    }

    /// GCP Vertex Gemini streaming API request
    async fn infer_stream<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
        }: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let mut request_body =
            serde_json::to_value(GCPVertexGeminiRequest::new(request, &self.model_id)?).map_err(
                |e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error serializing GCP Vertex Gemini request: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                },
            )?;
        let headers = inject_extra_request_data(
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            &mut request_body,
        )?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;

        let auth_headers = self
            .credentials
            .get_auth_headers(&self.audience, dynamic_api_keys)
            .await?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(&self.streaming_request_url)
            .json(&request_body)
            .headers(auth_headers)
            .headers(headers)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!(
                        "Error sending request to GCP Vertex Gemini: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    status_code: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
        let stream = stream_gcp_vertex_gemini(event_source, start_time).peekable();
        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        requests: &'a [ModelInferenceRequest<'_>],
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        let Some(batch_config) = &self.batch_config else {
            return Err(ErrorDetails::Config {
                message: "Missing config section: `[provider_types.gcp_vertex_gemini.batch]`"
                    .to_string(),
            }
            .into());
        };

        let auth_headers = self
            .credentials
            .get_auth_headers(&self.audience, dynamic_api_keys)
            .await?;

        let mut raw_requests = Vec::with_capacity(requests.len());
        let mut jsonl_data = Vec::new();
        for request in requests {
            let body = GCPVertexGeminiRequest::new(request, &self.model_id)?;
            let line =
                serde_json::to_string(&GCPVertexBatchLine { request: body }).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error serializing request: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                })?;

            jsonl_data.write_all(line.as_bytes()).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Error writing to JSONL: {}", DisplayOrDebugGateway::new(e)),
                })
            })?;
            jsonl_data.write_all(b"\n").map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Error writing to JSONL: {}", DisplayOrDebugGateway::new(e)),
                })
            })?;
            raw_requests.push(line);
        }

        let batch_id = Uuid::now_v7();
        let input_source_url = join_cloud_paths(
            &batch_config.input_uri_prefix,
            &format!("tensorzero-batch-input-{batch_id}.jsonl"),
        );

        // For now, we use the same set of credentials for writing to Google Cloud Storage as we do for invoking
        // the Vertex API. In the future, we may want to allow configuring a separate set of credentials.
        let store_and_path =
            make_gcp_object_store(&input_source_url, &self.credentials, dynamic_api_keys).await?;

        store_and_path
            .store
            .put(&store_and_path.path, jsonl_data.into())
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error uploading JSONL to object store: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;

        let request_body = GCPVertexGeminiBatchRequest {
            display_name: format!("tensorzero-batch-{batch_id}"),
            model: format!("publishers/google/models/{}", self.model_id.clone()),
            input_config: GCPVertexGeminiBatchRequestInputConfig::Jsonl {
                gcs_source: GCPVertexGeminiGCSSource {
                    uris: input_source_url,
                },
            },
            output_config: GCPVertexGeminiBatchRequestOutputConfig::Jsonl {
                gcs_destination: GCPVertexGeminiGCSDestination {
                    output_uri_prefix: join_cloud_paths(
                        &batch_config.output_uri_prefix,
                        &format!("tensorzero-batch-output-{batch_id}"),
                    ),
                },
            },
        };

        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;

        let res = http_client
            .post(batch_config.batch_request_url.clone())
            .headers(auth_headers)
            .body(raw_request.clone())
            .header(http::header::CONTENT_TYPE, "application/json")
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!("Error sending request: {}", DisplayOrDebugGateway::new(e)),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;

        if !res.status().is_success() {
            let response_code = res.status();
            let error_body = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error getting error response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;
            return Err(handle_gcp_vertex_gemini_error(
                raw_request.clone(),
                response_code,
                error_body,
            ));
        }

        let raw_response = res.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Error retrieving batch response: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                raw_request: Some(raw_request.clone()),
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;

        let response =
            serde_json::from_str::<GCPVertexBatchResponse>(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {raw_response}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                })
            })?;

        let batch_params = GCPVertexBatchParams {
            job_url_suffix: response.name,
        };

        Ok(StartBatchProviderInferenceResponse {
            batch_id,
            batch_params: serde_json::to_value(batch_params).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing batch params: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?,
            raw_requests,
            raw_request,
            raw_response,
            status: BatchStatus::Pending,
            errors: Vec::new(),
        })
    }

    async fn poll_batch_inference<'a>(
        &'a self,
        batch_request: &'a BatchRequestRow<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        let auth_headers = self
            .credentials
            .get_auth_headers(&self.audience, dynamic_api_keys)
            .await?;

        let batch_params: GCPVertexBatchParams = serde_json::from_value(
            batch_request.batch_params.clone().into_owned(),
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error deserializing batch params: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;

        let job_poll_url = self
            .api_v1_base_url
            .join(&batch_params.job_url_suffix)
            .map_err(|e| {
                Error::new(ErrorDetails::InternalError {
                    message: format!(
                        "Failed to join batch job URL - this should never happen: {e}"
                    ),
                })
            })?;

        let raw_request = job_poll_url.to_string();

        let res = http_client
            .get(job_poll_url)
            .headers(auth_headers)
            .header(http::header::CONTENT_TYPE, "application/json")
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!("Error sending request: {}", DisplayOrDebugGateway::new(e)),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;

        if !res.status().is_success() {
            let response_code = res.status();
            let error_body = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error getting error response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;
            return Err(handle_gcp_vertex_gemini_error(
                raw_request.clone(),
                response_code,
                error_body,
            ));
        }
        let raw_response = res.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Error retrieving batch response: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                raw_request: Some(raw_request.clone()),
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;
        let response =
            serde_json::from_str::<GCPVertexBatchResponse>(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {raw_response}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                })
            })?;
        match response.state {
            GCPVertexJobState::Pending
            | GCPVertexJobState::Running
            | GCPVertexJobState::Queued
            | GCPVertexJobState::Paused
            | GCPVertexJobState::Updating
            | GCPVertexJobState::Unspecified => Ok(PollBatchInferenceResponse::Pending {
                raw_request,
                raw_response,
            }),
            GCPVertexJobState::Succeeded | GCPVertexJobState::PartiallySucceeded => {
                let output_info = response.output_info.ok_or_else(|| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!(
                            "GCP Vertex Gemini batch response has no output info in state {:?}",
                            response.state
                        ),
                        raw_request: Some(raw_request.clone()),
                        raw_response: Some(raw_response.clone()),
                        provider_type: PROVIDER_TYPE.to_string(),
                    })
                })?;
                let batch_response = self
                    .collect_finished_batch(
                        output_info,
                        raw_request,
                        raw_response,
                        &self.credentials,
                        dynamic_api_keys,
                        batch_request,
                    )
                    .await?;
                Ok(PollBatchInferenceResponse::Completed(batch_response))
            }
            GCPVertexJobState::Failed
            | GCPVertexJobState::Cancelling
            | GCPVertexJobState::Expired
            | GCPVertexJobState::Cancelled
            | GCPVertexJobState::Unknown => Ok(PollBatchInferenceResponse::Failed {
                raw_request,
                raw_response,
            }),
        }
    }
}

fn stream_gcp_vertex_gemini(
    mut event_source: EventSource,
    start_time: Instant,
) -> ProviderInferenceResponseStreamInner {
    Box::pin(async_stream::stream! {
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    if matches!(e, reqwest_eventsource::Error::StreamEnded) {
                        break;
                    }
                    yield Err(convert_stream_error(PROVIDER_TYPE.to_string(), e).await);
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        let data: Result<GCPVertexGeminiResponse, Error> = serde_json::from_str(&message.data).map_err(|e| {
                            Error::new(ErrorDetails::InferenceServer {
                                message: format!("Error parsing streaming JSON response: {}", DisplayOrDebugGateway::new(e)),
                                provider_type: PROVIDER_TYPE.to_string(),
                                raw_request: None,
                                raw_response: Some(message.data.clone()),
                            })
                        });
                        let data = match data {
                            Ok(data) => data,
                            Err(e) => {
                                yield Err(e);
                                continue;
                            }
                        };
                        let response = GCPVertexGeminiStreamResponseWithMetadata {
                            raw_response: message.data,
                            response: data,
                            latency: start_time.elapsed(),

                        }.try_into();
                        yield response
                    }
                }
            }
         }
    })
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum GCPVertexGeminiRole {
    User,
    Model,
}

impl From<Role> for GCPVertexGeminiRole {
    fn from(role: Role) -> Self {
        match role {
            Role::User => GCPVertexGeminiRole::User,
            Role::Assistant => GCPVertexGeminiRole::Model,
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
struct GCPVertexGeminiFunctionCall<'a> {
    name: &'a str,
    args: Value,
}

#[derive(Debug, PartialEq, Serialize)]
struct GCPVertexGeminiFunctionResponse<'a> {
    name: &'a str,
    response: Value,
}

#[derive(Debug, PartialEq, Serialize)]
struct GCPVertexInlineData<'a> {
    mime_type: String,
    data: &'a str,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", untagged)]
enum GCPVertexGeminiContentPart<'a> {
    Text {
        text: &'a str,
    },
    InlineData {
        #[serde(rename = "inline_data")]
        inline_data: GCPVertexInlineData<'a>,
    },
    // TODO (if needed): FileData { file_data: FileData },
    FunctionCall {
        function_call: GCPVertexGeminiFunctionCall<'a>,
    },
    FunctionResponse {
        function_response: GCPVertexGeminiFunctionResponse<'a>,
    },
    // TODO (if needed): VideoMetadata { video_metadata: VideoMetadata },
}

impl<'a> TryFrom<&'a ContentBlock> for Option<FlattenUnknown<'a, GCPVertexGeminiContentPart<'a>>> {
    type Error = Error;

    fn try_from(block: &'a ContentBlock) -> Result<Self, Error> {
        match block {
            ContentBlock::Text(Text { text }) => Ok(Some(FlattenUnknown::Normal(
                GCPVertexGeminiContentPart::Text { text },
            ))),
            ContentBlock::ToolResult(tool_result) => {
                // GCP expects the format below according to [the documentation](https://ai.google.dev/gemini-api/docs/function-calling#multi-turn-example-1)
                let response = serde_json::json!({
                    "name": tool_result.name,
                    "content": tool_result.result
                });

                Ok(Some(FlattenUnknown::Normal(
                    GCPVertexGeminiContentPart::FunctionResponse {
                        function_response: GCPVertexGeminiFunctionResponse {
                            name: &tool_result.name,
                            response,
                        },
                    },
                )))
            }
            ContentBlock::ToolCall(tool_call) => {
                // Convert the tool call arguments from String to JSON Value (GCP expects an object)
                let args: Value = serde_json::from_str(&tool_call.arguments).map_err(|e| {
                    Error::new(ErrorDetails::InferenceClient {
                        status_code: Some(StatusCode::BAD_REQUEST),
                        message: format!(
                            "Error parsing tool call arguments as JSON Value: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: None,
                        raw_response: Some(tool_call.arguments.clone()),
                    })
                })?;

                if !args.is_object() {
                    return Err(ErrorDetails::InferenceClient {
                        status_code: Some(StatusCode::BAD_REQUEST),
                        message: "Tool call arguments must be a JSON object".to_string(),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: None,
                        raw_response: Some(tool_call.arguments.clone()),
                    }
                    .into());
                };

                Ok(Some(FlattenUnknown::Normal(
                    GCPVertexGeminiContentPart::FunctionCall {
                        function_call: GCPVertexGeminiFunctionCall {
                            name: &tool_call.name,
                            args,
                        },
                    },
                )))
            }
            ContentBlock::File(FileWithPath {
                file,
                storage_path: _,
            }) => {
                // All of our FileKinds are supported by GCP Vertex Gemini
                // If we add more, make sure to check their docs to see if they support it.
                match file.mime_type {
                    FileKind::Png | FileKind::Jpeg | FileKind::WebP | FileKind::Pdf => {}
                }
                Ok(Some(FlattenUnknown::Normal(
                    GCPVertexGeminiContentPart::InlineData {
                        inline_data: GCPVertexInlineData {
                            mime_type: file.mime_type.to_string(),
                            data: file.data()?.as_str(),
                        },
                    },
                )))
            }
            // We don't support thought blocks being passed in from a request.
            // These are only possible to be passed in in the scenario where the
            // output of a chat completion is used as an input to another model inference,
            // i.e. a judge or something.
            // We don't think the thoughts should be passed in in this case.
            ContentBlock::Thought(_thought) => Ok(None),
            ContentBlock::Unknown {
                data,
                model_provider_name: _,
            } => Ok(Some(FlattenUnknown::Unknown(Cow::Borrowed(data)))),
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
struct GCPVertexGeminiContent<'a> {
    role: GCPVertexGeminiRole,
    parts: Vec<FlattenUnknown<'a, GCPVertexGeminiContentPart<'a>>>,
}

impl<'a> TryFrom<&'a RequestMessage> for GCPVertexGeminiContent<'a> {
    type Error = Error;

    fn try_from(message: &'a RequestMessage) -> Result<Self, Error> {
        let role = GCPVertexGeminiRole::from(message.role);
        let parts: Vec<FlattenUnknown<GCPVertexGeminiContentPart>> = message
            .content
            .iter()
            .map(|block| block.try_into())
            .collect::<Result<Vec<Option<FlattenUnknown<GCPVertexGeminiContentPart>>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        Ok(GCPVertexGeminiContent { role, parts })
    }
}

#[derive(Debug, PartialEq, Serialize)]
struct GCPVertexGeminiFunctionDeclaration<'a> {
    name: &'a str,
    description: Option<&'a str>,
    parameters: Option<Value>, // Should be a JSONSchema as a Value
}

// TODO (if needed): implement [Retrieval](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/Tool#Retrieval)
// and [GoogleSearchRetrieval](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/Tool#GoogleSearchRetrieval)
// tools.
#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
enum GCPVertexGeminiTool<'a> {
    FunctionDeclarations(Vec<GCPVertexGeminiFunctionDeclaration<'a>>),
}

impl<'a> From<&'a ToolConfig> for GCPVertexGeminiFunctionDeclaration<'a> {
    fn from(tool: &'a ToolConfig) -> Self {
        let mut parameters = tool.parameters().clone();
        if let Some(obj) = parameters.as_object_mut() {
            obj.remove("additionalProperties");
            obj.remove("$schema");
        }

        GCPVertexGeminiFunctionDeclaration {
            name: tool.name(),
            description: Some(tool.description()),
            parameters: Some(parameters),
        }
    }
}

impl<'a> From<&'a Vec<ToolConfig>> for GCPVertexGeminiTool<'a> {
    fn from(tools: &'a Vec<ToolConfig>) -> Self {
        let function_declarations: Vec<GCPVertexGeminiFunctionDeclaration<'a>> =
            tools.iter().map(|tc| tc.into()).collect();
        GCPVertexGeminiTool::FunctionDeclarations(function_declarations)
    }
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum GCPVertexGeminiFunctionCallingMode {
    Auto,
    Any,
    None,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiFunctionCallingConfig<'a> {
    mode: GCPVertexGeminiFunctionCallingMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_function_names: Option<Vec<&'a str>>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiToolConfig<'a> {
    function_calling_config: GCPVertexGeminiFunctionCallingConfig<'a>,
}

// Auto is the default mode where a tool could be called but it isn't required.
// Any is a mode where a tool is required and if allowed_function_names is Some it has to be from that list.
// See [the documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/function-calling) for details.
// If Vertex adds any models that *don't* support Any mode, we'll add them to the list,
// which will cause us to fall back to Auto
const MODELS_NOT_SUPPORTING_ANY_MODE: &[&str] = &[];

impl<'a> From<(&'a ToolChoice, &'a str)> for GCPVertexGeminiToolConfig<'a> {
    fn from(input: (&'a ToolChoice, &'a str)) -> Self {
        let (tool_choice, model_name) = input;
        match tool_choice {
            ToolChoice::None => GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::None,
                    allowed_function_names: None,
                },
            },
            ToolChoice::Auto => GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Auto,
                    allowed_function_names: None,
                },
            },
            ToolChoice::Required => {
                if MODELS_NOT_SUPPORTING_ANY_MODE.contains(&model_name) {
                    GCPVertexGeminiToolConfig {
                        function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                            mode: GCPVertexGeminiFunctionCallingMode::Auto,
                            allowed_function_names: None,
                        },
                    }
                } else {
                    GCPVertexGeminiToolConfig {
                        function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                            mode: GCPVertexGeminiFunctionCallingMode::Any,
                            allowed_function_names: None,
                        },
                    }
                }
            }
            ToolChoice::Specific(tool_name) => {
                if MODELS_NOT_SUPPORTING_ANY_MODE.contains(&model_name) {
                    GCPVertexGeminiToolConfig {
                        function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                            mode: GCPVertexGeminiFunctionCallingMode::Auto,
                            allowed_function_names: None,
                        },
                    }
                } else {
                    GCPVertexGeminiToolConfig {
                        function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                            mode: GCPVertexGeminiFunctionCallingMode::Any,
                            allowed_function_names: Some(vec![tool_name]),
                        },
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
enum GCPVertexGeminiResponseMimeType {
    #[serde(rename = "text/plain")]
    TextPlain,
    #[serde(rename = "application/json")]
    ApplicationJson,
}

// TODO (if needed): add the other options [here](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerationConfig)
#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiGenerationConfig<'a> {
    stop_sequences: Option<Vec<&'a str>>,
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
    top_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u32>,
    response_mime_type: Option<GCPVertexGeminiResponseMimeType>,
    response_schema: Option<Value>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiRequest<'a> {
    contents: Vec<GCPVertexGeminiContent<'a>>,
    tools: Option<Vec<GCPVertexGeminiTool<'a>>>,
    tool_config: Option<GCPVertexGeminiToolConfig<'a>>,
    generation_config: Option<GCPVertexGeminiGenerationConfig<'a>>,
    system_instruction: Option<GCPVertexGeminiContent<'a>>,
    labels: HashMap<String, String>,
    // TODO (if needed): [Safety Settings](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/SafetySetting)
}

impl<'a> GCPVertexGeminiRequest<'a> {
    pub fn new(request: &'a ModelInferenceRequest<'a>, model_name: &'a str) -> Result<Self, Error> {
        if request.messages.is_empty() {
            return Err(ErrorDetails::InvalidRequest {
                message: "GCP Vertex Gemini requires at least one message".to_string(),
            }
            .into());
        }
        let system_instruction =
            request
                .system
                .as_ref()
                .map(|system_instruction| GCPVertexGeminiContentPart::Text {
                    text: system_instruction,
                });
        let contents: Vec<GCPVertexGeminiContent> = request
            .messages
            .iter()
            .map(GCPVertexGeminiContent::try_from)
            .filter_ok(|m| !m.parts.is_empty())
            .collect::<Result<_, _>>()?;
        let (tools, tool_config) = prepare_tools(request, model_name);
        let (response_mime_type, response_schema) = match request.json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                match request.output_schema {
                    Some(output_schema) => (
                        Some(GCPVertexGeminiResponseMimeType::ApplicationJson),
                        Some(process_output_schema(output_schema)?),
                    ),
                    None => (Some(GCPVertexGeminiResponseMimeType::ApplicationJson), None),
                }
            }
            ModelInferenceRequestJsonMode::Off => (None, None),
        };
        let generation_config = Some(GCPVertexGeminiGenerationConfig {
            stop_sequences: None,
            temperature: request.temperature,
            max_output_tokens: request.max_tokens,
            seed: request.seed,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            response_mime_type,
            response_schema,
        });
        Ok(GCPVertexGeminiRequest {
            contents,
            tools,
            tool_config,
            generation_config,
            system_instruction: system_instruction.map(|content| GCPVertexGeminiContent {
                role: GCPVertexGeminiRole::Model,
                parts: vec![FlattenUnknown::Normal(content)],
            }),
            // We attach our custom tag so that we can identify the original inference when
            // retrieving batch results.
            labels: [(
                INFERENCE_ID_LABEL.to_string(),
                request.inference_id.to_string(),
            )]
            .into_iter()
            .collect(),
        })
    }
}

fn prepare_tools<'a>(
    request: &'a ModelInferenceRequest<'a>,
    model_name: &'a str,
) -> (
    Option<Vec<GCPVertexGeminiTool<'a>>>,
    Option<GCPVertexGeminiToolConfig<'a>>,
) {
    match &request.tool_config {
        Some(tool_config) => {
            if tool_config.tools_available.is_empty() {
                return (None, None);
            }
            let tools = Some(vec![(&tool_config.tools_available).into()]);
            let tool_config = Some((&tool_config.tool_choice, model_name).into());
            (tools, tool_config)
        }
        None => (None, None),
    }
}

pub(crate) fn process_output_schema(output_schema: &Value) -> Result<Value, Error> {
    let mut schema = output_schema.clone();

    /// Recursively remove all instances of "additionalProperties" and "$schema"
    fn remove_properties(value: &mut Value) {
        match value {
            Value::Object(obj) => {
                obj.remove("additionalProperties");
                obj.remove("$schema");
                for (_, v) in obj.iter_mut() {
                    remove_properties(v);
                }
            }
            Value::Array(arr) => {
                for v in arr.iter_mut() {
                    remove_properties(v);
                }
            }
            _ => {}
        }
    }

    remove_properties(&mut schema);
    Ok(schema)
}

#[derive(Debug, Deserialize, Serialize)]
struct GCPVertexGeminiResponseFunctionCall {
    name: String,
    args: Value,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
enum GCPVertexGeminiResponseContentPart {
    Text(String),
    // TODO (if needed): InlineData { inline_data: Blob },
    // TODO (if needed): FileData { file_data: FileData },
    FunctionCall(GCPVertexGeminiResponseFunctionCall),
    ExecutableCode(serde_json::Value),
    // TODO (if needed): FunctionResponse
    // TODO (if needed): VideoMetadata { video_metadata: VideoMetadata },
}

impl TryFrom<GCPVertexGeminiResponseContentPart> for ContentBlockChunk {
    type Error = Error;
    /// GCP Vertex Gemini does not support parallel tool calling or multiple content blocks as far as I can tell.
    /// So there is no issue with bookkeeping IDs for content blocks.
    /// We should revisit this if they begin to support it.
    fn try_from(part: GCPVertexGeminiResponseContentPart) -> Result<Self, Self::Error> {
        match part {
            GCPVertexGeminiResponseContentPart::Text(text) => Ok(ContentBlockChunk::Text(TextChunk {
                text,
                id: "0".to_string(),
            })),
            GCPVertexGeminiResponseContentPart::FunctionCall(function_call) => {
                let arguments = serialize_or_log(&function_call.args);
                Ok(ContentBlockChunk::ToolCall(ToolCallChunk {
                    raw_name: function_call.name,
                    raw_arguments: arguments,
                    id: "0".to_string(),
                }))
            }
            GCPVertexGeminiResponseContentPart::ExecutableCode(_) => {
                Err(Error::new(ErrorDetails::InferenceServer {
                    message: "executableCode is not supported in streaming response for GCP Vertex Gemini".to_string(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: None,
                    raw_response: Some(serde_json::to_string(&part).unwrap_or_default()),
                }))
            }
        }
    }
}

fn convert_to_output(
    model_name: &str,
    provider_name: &str,
    part: GCPVertexGeminiResponseContentPart,
) -> Result<ContentBlockOutput, Error> {
    match part {
        GCPVertexGeminiResponseContentPart::Text(text) => Ok(text.into()),
        GCPVertexGeminiResponseContentPart::FunctionCall(function_call) => {
            Ok(ContentBlockOutput::ToolCall(ToolCall {
                name: function_call.name,
                arguments: serde_json::to_string(&function_call.args).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error serializing function call arguments returned from GCP: {e}"
                        ),
                    })
                })?,
                // GCP doesn't have the concept of tool call ID so we generate one for our bookkeeping
                id: Uuid::now_v7().to_string(),
            }))
        }
        GCPVertexGeminiResponseContentPart::ExecutableCode(data) => {
            Ok(ContentBlockOutput::Unknown {
                data: serde_json::json!({
                    "executableCode": data,
                }),
                model_provider_name: Some(fully_qualified_name(model_name, provider_name)),
            })
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct GCPVertexGeminiResponseContent {
    parts: Vec<GCPVertexGeminiResponseContentPart>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum GCPVertexGeminiFinishReason {
    FinishReasonUnspecified,
    Stop,
    MaxTokens,
    Safety,
    Recitation,
    Other,
    Blocklist,
    ProhibitedContent,
    #[serde(rename = "SPII")]
    Spii,
    MalformedFunctionCall,
    #[serde(other)]
    Unknown,
}

impl From<GCPVertexGeminiFinishReason> for FinishReason {
    fn from(finish_reason: GCPVertexGeminiFinishReason) -> Self {
        match finish_reason {
            GCPVertexGeminiFinishReason::Stop => FinishReason::Stop,
            GCPVertexGeminiFinishReason::MaxTokens => FinishReason::Length,
            GCPVertexGeminiFinishReason::Safety => FinishReason::ContentFilter,
            GCPVertexGeminiFinishReason::Recitation => FinishReason::ToolCall,
            GCPVertexGeminiFinishReason::Other => FinishReason::Unknown,
            GCPVertexGeminiFinishReason::Blocklist => FinishReason::ContentFilter,
            GCPVertexGeminiFinishReason::ProhibitedContent => FinishReason::ContentFilter,
            GCPVertexGeminiFinishReason::Spii => FinishReason::ContentFilter,
            GCPVertexGeminiFinishReason::MalformedFunctionCall => FinishReason::ToolCall,
            GCPVertexGeminiFinishReason::FinishReasonUnspecified => FinishReason::Unknown,
            GCPVertexGeminiFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiResponseCandidate {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<GCPVertexGeminiResponseContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<GCPVertexGeminiFinishReason>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiUsageMetadata {
    prompt_token_count: Option<u32>,
    // GCP doesn't return output tokens in certain edge cases (e.g. generation blocked by safety settings)
    #[serde(skip_serializing_if = "Option::is_none")]
    candidates_token_count: Option<u32>,
}

impl From<GCPVertexGeminiUsageMetadata> for Usage {
    fn from(usage_metadata: GCPVertexGeminiUsageMetadata) -> Self {
        Usage {
            input_tokens: usage_metadata.prompt_token_count.unwrap_or(0),
            output_tokens: usage_metadata.candidates_token_count.unwrap_or(0),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiResponse {
    candidates: Vec<GCPVertexGeminiResponseCandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage_metadata: Option<GCPVertexGeminiUsageMetadata>,
}

struct GCPVertexGeminiResponseWithMetadata<'a> {
    response: GCPVertexGeminiResponse,
    raw_response: String,
    latency: Latency,
    request: serde_json::Value,
    generic_request: &'a ModelInferenceRequest<'a>,
    model_name: &'a str,
    provider_name: &'a str,
}

fn get_response_content(
    response: GCPVertexGeminiResponse,
    raw_request: &str,
    raw_response: &str,
    model_name: &str,
    provider_name: &str,
) -> Result<(Vec<ContentBlockOutput>, Option<FinishReason>), Error> {
    // GCP Vertex Gemini response can contain multiple candidates and each of these can contain
    // multiple content parts. We will only use the first candidate but handle all parts of the response therein.
    let first_candidate = response.candidates.into_iter().next().ok_or_else(|| {
        Error::new(ErrorDetails::InferenceServer {
            message: "GCP Vertex Gemini response has no candidates".to_string(),
            raw_request: Some(raw_request.to_string()),
            raw_response: Some(raw_response.to_string()),
            provider_type: PROVIDER_TYPE.to_string(),
        })
    })?;

    let finish_reason = first_candidate
        .finish_reason
        .map(|finish_reason| finish_reason.into());

    // GCP sometimes doesn't return content in the response (e.g. safety settings blocked the generation).
    let content = match first_candidate.content {
        Some(content) => content
            .parts
            .into_iter()
            .map(|part| convert_to_output(model_name, provider_name, part))
            .collect::<Result<Vec<ContentBlockOutput>, Error>>()?,
        None => vec![],
    };
    Ok((content, finish_reason))
}

impl<'a> TryFrom<GCPVertexGeminiResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(response: GCPVertexGeminiResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let GCPVertexGeminiResponseWithMetadata {
            response,
            raw_response,
            latency,
            request: request_body,
            generic_request,
            model_name,
            provider_name,
        } = response;

        let usage = response
            .usage_metadata
            .clone()
            .ok_or_else(|| {
                Error::new(ErrorDetails::InferenceServer {
                    message: "GCP Vertex Gemini non-streaming response has no usage metadata"
                        .to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?
            .into();
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let system = generic_request.system.clone();
        let input_messages = generic_request.messages.clone();

        let (content, finish_reason) = get_response_content(
            response,
            &raw_request,
            &raw_response,
            model_name,
            provider_name,
        )?;

        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages,
                raw_request,
                raw_response,
                usage,
                latency,
                finish_reason,
            },
        ))
    }
}

struct GCPVertexGeminiStreamResponseWithMetadata {
    raw_response: String,
    response: GCPVertexGeminiResponse,
    latency: Duration,
}

impl TryFrom<GCPVertexGeminiStreamResponseWithMetadata> for ProviderInferenceResponseChunk {
    type Error = Error;
    fn try_from(response: GCPVertexGeminiStreamResponseWithMetadata) -> Result<Self, Self::Error> {
        let GCPVertexGeminiStreamResponseWithMetadata {
            response,
            latency,
            raw_response,
        } = response;

        let first_candidate = response.candidates.into_iter().next().ok_or_else(|| {
            Error::new(ErrorDetails::InferenceServer {
                message: "GCP Vertex Gemini response has no candidates".to_string(),
                raw_request: None,
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;

        // GCP sometimes returns chunks without content (e.g. they might have usage only).
        let mut content: Vec<ContentBlockChunk> = match first_candidate.content {
            Some(content) => content
                .parts
                .into_iter()
                .map(|part| part.try_into())
                .collect::<Result<Vec<ContentBlockChunk>, Error>>()?,
            None => vec![],
        };

        // GCP occasionally spuriously returns empty text chunks. We filter these out.
        content.retain(|chunk| match chunk {
            ContentBlockChunk::Text(text) => !text.text.is_empty(),
            _ => true,
        });
        Ok(ProviderInferenceResponseChunk::new(
            content,
            response
                .usage_metadata
                .map(|usage_metadata| usage_metadata.into()),
            raw_response,
            latency,
            first_candidate
                .finish_reason
                .map(|finish_reason| finish_reason.into()),
        ))
    }
}

fn handle_gcp_vertex_gemini_error(
    raw_request: String,
    response_code: StatusCode,
    response_body: String,
) -> Error {
    match response_code {
        StatusCode::UNAUTHORIZED
        | StatusCode::BAD_REQUEST
        | StatusCode::PAYLOAD_TOO_LARGE
        | StatusCode::TOO_MANY_REQUESTS => Error::new(ErrorDetails::InferenceClient {
            message: response_body.clone(),
            status_code: Some(response_code),
            raw_request: Some(raw_request),
            raw_response: Some(response_body.clone()),
            provider_type: PROVIDER_TYPE.to_string(),
        }),
        // StatusCode::NOT_FOUND | StatusCode::FORBIDDEN | StatusCode::INTERNAL_SERVER_ERROR | 529: Overloaded
        // These are all captured in _ since they have the same error behavior
        _ => Error::new(ErrorDetails::InferenceServer {
            message: response_body.clone(),
            raw_request: Some(raw_request),
            raw_response: Some(response_body.clone()),
            provider_type: PROVIDER_TYPE.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use serde_json::json;

    use super::*;
    use crate::inference::providers::test_helpers::{MULTI_TOOL_CONFIG, QUERY_TOOL, WEATHER_TOOL};
    use crate::inference::types::{FunctionType, ModelInferenceRequestJsonMode};
    use crate::tool::{ToolCallConfig, ToolResult};

    #[test]
    fn test_gcp_vertex_content_try_from() {
        let message = RequestMessage {
            role: Role::User,
            content: vec!["Hello, world!".to_string().into()],
        };
        let content = GCPVertexGeminiContent::try_from(&message).unwrap();
        assert_eq!(content.role, GCPVertexGeminiRole::User);
        assert_eq!(content.parts.len(), 1);
        assert_eq!(
            content.parts[0],
            FlattenUnknown::Normal(GCPVertexGeminiContentPart::Text {
                text: "Hello, world!"
            })
        );

        let message = RequestMessage {
            role: Role::Assistant,
            content: vec!["Hello, world!".to_string().into()],
        };
        let content = GCPVertexGeminiContent::try_from(&message).unwrap();
        assert_eq!(content.role, GCPVertexGeminiRole::Model);
        assert_eq!(content.parts.len(), 1);
        assert_eq!(
            content.parts[0],
            FlattenUnknown::Normal(GCPVertexGeminiContentPart::Text {
                text: "Hello, world!"
            })
        );
        let message = RequestMessage {
            role: Role::Assistant,
            content: vec![
                "Here's the result of the function call:".to_string().into(),
                ContentBlock::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    name: "get_temperature".to_string(),
                    arguments: r#"{"location": "New York", "unit": "celsius"}"#.to_string(),
                }),
            ],
        };
        let content = GCPVertexGeminiContent::try_from(&message).unwrap();
        assert_eq!(content.role, GCPVertexGeminiRole::Model);
        assert_eq!(content.parts.len(), 2);
        assert_eq!(
            content.parts[0],
            FlattenUnknown::Normal(GCPVertexGeminiContentPart::Text {
                text: "Here's the result of the function call:"
            })
        );
        assert_eq!(
            content.parts[1],
            FlattenUnknown::Normal(GCPVertexGeminiContentPart::FunctionCall {
                function_call: GCPVertexGeminiFunctionCall {
                    name: "get_temperature",
                    args: json!({"location": "New York", "unit": "celsius"}),
                }
            })
        );

        let message = RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                id: "call_1".to_string(),
                name: "get_temperature".to_string(),
                result: r#"{"temperature": 25, "conditions": "sunny"}"#.to_string(),
            })],
        };
        let content = GCPVertexGeminiContent::try_from(&message).unwrap();
        assert_eq!(content.role, GCPVertexGeminiRole::User);
        assert_eq!(content.parts.len(), 1);
        assert_eq!(
            content.parts[0],
            FlattenUnknown::Normal(GCPVertexGeminiContentPart::FunctionResponse {
                function_response: GCPVertexGeminiFunctionResponse {
                    name: "get_temperature",
                    response: json!({
                        "name": "get_temperature",
                        "content": r#"{"temperature": 25, "conditions": "sunny"}"#
                    }),
                }
            })
        );
    }

    #[test]
    fn test_from_vec_tool() {
        let tool = GCPVertexGeminiTool::from(&MULTI_TOOL_CONFIG.tools_available);
        assert_eq!(
            tool,
            GCPVertexGeminiTool::FunctionDeclarations(vec![
                GCPVertexGeminiFunctionDeclaration {
                    name: "get_temperature",
                    description: Some("Get the current temperature in a given location"),
                    parameters: Some(MULTI_TOOL_CONFIG.tools_available[0].parameters().clone()),
                },
                GCPVertexGeminiFunctionDeclaration {
                    name: "query_articles",
                    description: Some("Query articles from Wikipedia"),
                    parameters: Some(MULTI_TOOL_CONFIG.tools_available[1].parameters().clone()),
                }
            ])
        );
    }

    #[test]
    fn test_from_tool_choice() {
        let tool_choice = ToolChoice::Auto;
        let supports_any_model_name = "gemini-2.5-pro-preview-05-06";
        let tool_config = GCPVertexGeminiToolConfig::from((&tool_choice, supports_any_model_name));
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Auto,
                    allowed_function_names: None,
                }
            }
        );

        // The Pro model supports Any mode
        let tool_choice = ToolChoice::Required;
        let tool_config = GCPVertexGeminiToolConfig::from((&tool_choice, supports_any_model_name));
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Any,
                    allowed_function_names: None,
                }
            }
        );

        // The Pro model supports Any mode with allowed function names
        let tool_choice = ToolChoice::Specific("get_temperature".to_string());
        let tool_config = GCPVertexGeminiToolConfig::from((&tool_choice, supports_any_model_name));
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Any,
                    allowed_function_names: Some(vec!["get_temperature"]),
                }
            }
        );

        let tool_choice = ToolChoice::None;
        let tool_config = GCPVertexGeminiToolConfig::from((&tool_choice, supports_any_model_name));
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::None,
                    allowed_function_names: None,
                }
            }
        );
    }

    #[test]
    fn test_gcp_vertex_request_try_from() {
        // Test Case 1: Empty message list
        let tool_config = ToolCallConfig {
            tools_available: vec![],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
        };
        let inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let result = GCPVertexGeminiRequest::new(&inference_request, "gemini-pro");
        let details = result.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
                message: "GCP Vertex Gemini requires at least one message".to_string()
            }
        );

        // Test Case 2: Messages with System instructions
        let messages = vec![
            RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            },
        ];
        let inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: messages.clone(),
            system: Some("test_system".to_string()),
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let result = GCPVertexGeminiRequest::new(&inference_request, "gemini-pro");
        let request = result.unwrap();
        assert_eq!(request.contents.len(), 2);
        assert_eq!(request.contents[0].role, GCPVertexGeminiRole::User);
        assert_eq!(
            request.contents[0].parts[0],
            FlattenUnknown::Normal(GCPVertexGeminiContentPart::Text { text: "test_user" })
        );
        assert_eq!(request.contents[1].role, GCPVertexGeminiRole::Model);
        assert_eq!(request.contents[1].parts.len(), 1);
        assert_eq!(
            request.contents[1].parts[0],
            FlattenUnknown::Normal(GCPVertexGeminiContentPart::Text {
                text: "test_assistant"
            })
        );

        // Test case 3: Messages with system message and some of the optional fields are tested
        let messages = vec![
            RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            },
            RequestMessage {
                role: Role::User,
                content: vec!["test_user2".to_string().into()],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            },
        ];
        let output_schema = serde_json::json!({});
        let inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: messages.clone(),
            system: Some("test_system".to_string()),
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::On,
            function_type: FunctionType::Chat,
            output_schema: Some(&output_schema),
            extra_body: Default::default(),
            ..Default::default()
        };
        // JSON schema should be supported for Gemini Pro models
        let result =
            GCPVertexGeminiRequest::new(&inference_request, "gemini-2.5-pro-preview-05-06");
        let request = result.unwrap();
        assert_eq!(request.contents.len(), 3);
        assert_eq!(request.contents[0].role, GCPVertexGeminiRole::User);
        assert_eq!(request.contents[1].role, GCPVertexGeminiRole::User);
        assert_eq!(request.contents[2].role, GCPVertexGeminiRole::Model);
        assert_eq!(request.contents[0].parts.len(), 1);
        assert_eq!(request.contents[1].parts.len(), 1);
        assert_eq!(request.contents[2].parts.len(), 1);
        assert_eq!(
            request.contents[0].parts[0],
            FlattenUnknown::Normal(GCPVertexGeminiContentPart::Text { text: "test_user" })
        );
        assert_eq!(
            request.contents[1].parts[0],
            FlattenUnknown::Normal(GCPVertexGeminiContentPart::Text { text: "test_user2" })
        );
        assert_eq!(
            request.contents[2].parts[0],
            FlattenUnknown::Normal(GCPVertexGeminiContentPart::Text {
                text: "test_assistant"
            })
        );
        assert_eq!(
            request.generation_config.as_ref().unwrap().temperature,
            Some(0.5)
        );
        assert_eq!(
            request
                .generation_config
                .as_ref()
                .unwrap()
                .max_output_tokens,
            Some(100)
        );
        assert_eq!(request.generation_config.as_ref().unwrap().seed, Some(69));
        assert_eq!(
            request
                .generation_config
                .as_ref()
                .unwrap()
                .response_mime_type,
            Some(GCPVertexGeminiResponseMimeType::ApplicationJson)
        );
        assert_eq!(
            request.generation_config.as_ref().unwrap().response_schema,
            Some(output_schema.clone())
        );

        let inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: messages.clone(),
            system: Some("test_system".to_string()),
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::On,
            function_type: FunctionType::Chat,
            output_schema: Some(&output_schema),
            extra_body: Default::default(),
            ..Default::default()
        };
        // JSON mode should be supported for Gemini Flash models but without a schema
        let result = GCPVertexGeminiRequest::new(&inference_request, "gemini-flash");
        let request = result.unwrap();
        assert_eq!(request.contents.len(), 3);
        assert_eq!(request.contents[0].role, GCPVertexGeminiRole::User);
        assert_eq!(request.contents[1].role, GCPVertexGeminiRole::User);
        assert_eq!(request.contents[2].role, GCPVertexGeminiRole::Model);
        assert_eq!(request.contents[0].parts.len(), 1);
        assert_eq!(request.contents[1].parts.len(), 1);
        assert_eq!(request.contents[2].parts.len(), 1);
        assert_eq!(
            request.contents[0].parts[0],
            FlattenUnknown::Normal(GCPVertexGeminiContentPart::Text { text: "test_user" })
        );
        assert_eq!(
            request.contents[1].parts[0],
            FlattenUnknown::Normal(GCPVertexGeminiContentPart::Text { text: "test_user2" })
        );
        assert_eq!(
            request.contents[2].parts[0],
            FlattenUnknown::Normal(GCPVertexGeminiContentPart::Text {
                text: "test_assistant"
            })
        );
        assert_eq!(
            request.generation_config.as_ref().unwrap().temperature,
            Some(0.5)
        );
        assert_eq!(request.generation_config.as_ref().unwrap().top_p, Some(0.9));
        assert_eq!(
            request.generation_config.as_ref().unwrap().presence_penalty,
            Some(0.1)
        );
        assert_eq!(
            request
                .generation_config
                .as_ref()
                .unwrap()
                .frequency_penalty,
            Some(0.1)
        );
        assert_eq!(
            request
                .generation_config
                .as_ref()
                .unwrap()
                .max_output_tokens,
            Some(100)
        );
        assert_eq!(request.generation_config.as_ref().unwrap().seed, Some(69));
        assert_eq!(
            request
                .generation_config
                .as_ref()
                .unwrap()
                .response_mime_type,
            Some(GCPVertexGeminiResponseMimeType::ApplicationJson)
        );
        assert_eq!(
            request.generation_config.as_ref().unwrap().response_schema,
            Some(serde_json::Value::Object(Default::default()))
        );
    }

    #[test]
    fn test_gcp_to_t0_response() {
        let part = GCPVertexGeminiResponseContentPart::Text("test_assistant".to_string());
        let content = GCPVertexGeminiResponseContent { parts: vec![part] };
        let candidate = GCPVertexGeminiResponseCandidate {
            content: Some(content),
            finish_reason: Some(GCPVertexGeminiFinishReason::Stop),
        };
        let response = GCPVertexGeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GCPVertexGeminiUsageMetadata {
                prompt_token_count: None,
                candidates_token_count: None,
            }),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_secs(1),
        };
        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: Some("test_system".to_string()),
            tool_config: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let request_body = GCPVertexGeminiRequest {
            contents: vec![],
            generation_config: None,
            tools: None,
            tool_config: None,
            system_instruction: None,
            labels: HashMap::new(),
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let raw_response = "test response".to_string();
        let response_with_latency = GCPVertexGeminiResponseWithMetadata {
            response,
            latency: latency.clone(),
            request: serde_json::to_value(&request_body).unwrap(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
            model_name: "gemini-pro",
            provider_name: "gcp_vertex_gemini",
        };
        let model_inference_response: ProviderInferenceResponse =
            response_with_latency.try_into().unwrap();
        assert_eq!(
            model_inference_response.output,
            vec!["test_assistant".to_string().into()]
        );
        assert_eq!(
            model_inference_response.usage,
            Usage {
                input_tokens: 0,
                output_tokens: 0,
            }
        );
        assert_eq!(model_inference_response.latency, latency);
        assert_eq!(model_inference_response.raw_request, raw_request);
        assert_eq!(model_inference_response.raw_response, raw_response);
        assert_eq!(
            model_inference_response.finish_reason,
            Some(FinishReason::Stop)
        );
        assert_eq!(
            model_inference_response.system,
            Some("test_system".to_string())
        );
        assert_eq!(model_inference_response.input_messages, vec![]);
        let text_part =
            GCPVertexGeminiResponseContentPart::Text("Here's the weather information:".to_string());
        let function_call_part =
            GCPVertexGeminiResponseContentPart::FunctionCall(GCPVertexGeminiResponseFunctionCall {
                name: "get_temperature".to_string(),
                args: json!({"location": "New York", "unit": "celsius"}),
            });
        let content = GCPVertexGeminiResponseContent {
            parts: vec![text_part, function_call_part],
        };
        let candidate = GCPVertexGeminiResponseCandidate {
            content: Some(content),
            finish_reason: Some(GCPVertexGeminiFinishReason::Stop),
        };
        let response = GCPVertexGeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GCPVertexGeminiUsageMetadata {
                prompt_token_count: Some(15),
                candidates_token_count: Some(20),
            }),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_secs(2),
        };
        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }],
            system: None,
            tool_config: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let request_body = GCPVertexGeminiRequest {
            contents: vec![],
            generation_config: None,
            tools: None,
            tool_config: None,
            system_instruction: None,
            labels: HashMap::new(),
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let response_with_latency = GCPVertexGeminiResponseWithMetadata {
            response,
            latency: latency.clone(),
            request: serde_json::to_value(&request_body).unwrap(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
            model_name: "gemini-pro",
            provider_name: "gcp_vertex_gemini",
        };
        let model_inference_response: ProviderInferenceResponse =
            response_with_latency.try_into().unwrap();

        if let [ContentBlockOutput::Text(Text { text }), ContentBlockOutput::ToolCall(tool_call)] =
            &model_inference_response.output[..]
        {
            assert_eq!(text, "Here's the weather information:");
            assert_eq!(tool_call.name, "get_temperature");
            assert_eq!(
                tool_call.arguments,
                r#"{"location":"New York","unit":"celsius"}"#
            );
        } else {
            panic!("Expected a text and tool call content block");
        }

        assert_eq!(
            model_inference_response.usage,
            Usage {
                input_tokens: 15,
                output_tokens: 20,
            }
        );
        assert_eq!(model_inference_response.latency, latency);
        assert_eq!(
            model_inference_response.finish_reason,
            Some(FinishReason::Stop)
        );
        assert_eq!(model_inference_response.raw_request, raw_request);
        assert_eq!(model_inference_response.raw_response, raw_response);
        assert_eq!(model_inference_response.system, None);
        assert_eq!(
            model_inference_response.input_messages,
            vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }]
        );

        let text_part1 =
            GCPVertexGeminiResponseContentPart::Text("Here's the weather information:".to_string());
        let function_call_part =
            GCPVertexGeminiResponseContentPart::FunctionCall(GCPVertexGeminiResponseFunctionCall {
                name: "get_temperature".to_string(),
                args: json!({"location": "New York", "unit": "celsius"}),
            });
        let text_part2 = GCPVertexGeminiResponseContentPart::Text(
            "And here's a restaurant recommendation:".to_string(),
        );
        let function_call_part2 =
            GCPVertexGeminiResponseContentPart::FunctionCall(GCPVertexGeminiResponseFunctionCall {
                name: "get_restaurant".to_string(),
                args: json!({"cuisine": "Italian", "price_range": "moderate"}),
            });
        let content = GCPVertexGeminiResponseContent {
            parts: vec![
                text_part1,
                function_call_part,
                text_part2,
                function_call_part2,
            ],
        };
        let candidate = GCPVertexGeminiResponseCandidate {
            content: Some(content),
            finish_reason: Some(GCPVertexGeminiFinishReason::Stop),
        };
        let response = GCPVertexGeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GCPVertexGeminiUsageMetadata {
                prompt_token_count: Some(25),
                candidates_token_count: Some(40),
            }),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_secs(3),
        };
        let request_body = GCPVertexGeminiRequest {
            contents: vec![],
            generation_config: None,
            tools: None,
            tool_config: None,
            system_instruction: None,
            labels: HashMap::new(),
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let response_with_latency = GCPVertexGeminiResponseWithMetadata {
            response,
            latency: latency.clone(),
            request: serde_json::to_value(&request_body).unwrap(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
            model_name: "gemini-pro",
            provider_name: "gcp_vertex_gemini",
        };
        let model_inference_response: ProviderInferenceResponse =
            response_with_latency.try_into().unwrap();
        assert_eq!(model_inference_response.raw_request, raw_request);

        if let [ContentBlockOutput::Text(Text { text: text1 }), ContentBlockOutput::ToolCall(tool_call1), ContentBlockOutput::Text(Text { text: text2 }), ContentBlockOutput::ToolCall(tool_call2)] =
            &model_inference_response.output[..]
        {
            assert_eq!(text1, "Here's the weather information:");
            assert_eq!(text2, "And here's a restaurant recommendation:");
            assert_eq!(tool_call1.name, "get_temperature");
            assert_eq!(
                tool_call1.arguments,
                r#"{"location":"New York","unit":"celsius"}"#
            );
            assert_eq!(tool_call2.name, "get_restaurant");
            assert_eq!(
                tool_call2.arguments,
                r#"{"cuisine":"Italian","price_range":"moderate"}"#
            );
        } else {
            panic!(
                "Content does not match expected structure: {:?}",
                model_inference_response.output
            );
        }

        assert_eq!(
            model_inference_response.usage,
            Usage {
                input_tokens: 25,
                output_tokens: 40,
            }
        );
        assert_eq!(model_inference_response.latency, latency);
        assert_eq!(model_inference_response.raw_request, raw_request);
        assert_eq!(model_inference_response.raw_response, raw_response);
        assert_eq!(model_inference_response.system, None);
        assert_eq!(
            model_inference_response.input_messages,
            vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }]
        );
    }

    #[test]
    fn test_prepare_tools() {
        let request_with_tools = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&MULTI_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let (tools, tool_choice) =
            prepare_tools(&request_with_tools, "gemini-2.5-pro-preview-05-06");
        let tools = tools.unwrap();
        let tool_config = tool_choice.unwrap();
        assert_eq!(
            tool_config.function_calling_config.mode,
            GCPVertexGeminiFunctionCallingMode::Any,
        );
        assert_eq!(tools.len(), 1);
        match &tools[0] {
            GCPVertexGeminiTool::FunctionDeclarations(function_declarations) => {
                assert_eq!(function_declarations.len(), 2);
                assert_eq!(function_declarations[0].name, WEATHER_TOOL.name());
                assert_eq!(
                    function_declarations[0].parameters,
                    Some(WEATHER_TOOL.parameters().clone())
                );
                assert_eq!(function_declarations[1].name, QUERY_TOOL.name());
                assert_eq!(
                    function_declarations[1].parameters,
                    Some(QUERY_TOOL.parameters().clone())
                );
            }
        }
        let request_with_tools = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&MULTI_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let (tools, tool_choice) = prepare_tools(&request_with_tools, "gemini-2.0-flash-lite");
        let tools = tools.unwrap();
        let tool_config = tool_choice.unwrap();
        assert_eq!(
            tool_config.function_calling_config.mode,
            GCPVertexGeminiFunctionCallingMode::Any,
        );
        assert_eq!(tools.len(), 1);
        match &tools[0] {
            GCPVertexGeminiTool::FunctionDeclarations(function_declarations) => {
                assert_eq!(function_declarations.len(), 2);
                assert_eq!(function_declarations[0].name, WEATHER_TOOL.name());
                assert_eq!(
                    function_declarations[0].parameters,
                    Some(WEATHER_TOOL.parameters().clone())
                );
                assert_eq!(function_declarations[1].name, QUERY_TOOL.name());
                assert_eq!(
                    function_declarations[1].parameters,
                    Some(QUERY_TOOL.parameters().clone())
                );
            }
        }
    }

    #[test]
    fn test_process_output_schema() {
        let output_schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "email": {"type": "string", "format": "email"}
            }
        });
        let processed_schema = process_output_schema(&output_schema).unwrap();
        assert_eq!(processed_schema, output_schema);

        // Test with a schema that includes additionalProperties
        let output_schema_with_additional = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
            "additionalProperties": true
        });
        let output_schema_without_additional = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
        });
        let processed_schema_with_additional =
            process_output_schema(&output_schema_with_additional).unwrap();
        assert_eq!(
            processed_schema_with_additional,
            output_schema_without_additional
        );

        // Test with a schema that explicitly disallows additional properties
        let output_schema_no_additional = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
            "additionalProperties": false
        });
        let processed_schema_no_additional =
            process_output_schema(&output_schema_no_additional).unwrap();
        assert_eq!(
            processed_schema_no_additional,
            output_schema_without_additional
        );

        // Test with a schema that includes recursive additionalProperties
        let output_schema_recursive = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "children": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer", "minimum": 0}
                        },
                        "additionalProperties": {
                            "$ref": "#"
                        }
                    }
                }
            },
            "additionalProperties": {
                "$ref": "#"
            }
        });
        let expected_processed_schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "children": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer", "minimum": 0}
                        }
                    }
                }
            }
        });
        let processed_schema_recursive = process_output_schema(&output_schema_recursive).unwrap();
        assert_eq!(processed_schema_recursive, expected_processed_schema);

        // Test with schema containing $schema at top level and in child objects
        let output_schema_with_schema_fields = json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nested": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"}
                    }
                },
                "array": {
                    "type": "array",
                    "items": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "string"
                    }
                }
            }
        });
        let expected_schema_without_schema_fields = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nested": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"}
                    }
                },
                "array": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            }
        });
        let processed_schema = process_output_schema(&output_schema_with_schema_fields).unwrap();
        assert_eq!(processed_schema, expected_schema_without_schema_fields);
    }

    #[test]
    fn test_credential_to_gcp_vertex_credentials() {
        // Test valid JSON file contents using the sample from dev guide
        let json_content = r#"{
            "type": "service_account",
            "project_id": "none",
            "private_key_id": "none",
            "private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIICXAIBAAKBgQDAKxbF0dfne7PmPwpFEcSi2JFBeO98DXW7bimAPE6dHHCkDvoU\nlD/fy8svrPU6xsCYxM3LfKY/F+s/P+FizXUQ6eDu5ipYCRfweiQ4gqms+zROeORA\nJez3zelPQ7vY/MYCnp0LYYCH2HTyBeMFIX+Rgwjral495j0O6uV7cjgneQIDAQAB\nAoGAOXcpMjLUS6bUX1AOtCTiFoiIt3mAtCoaQNhqlKx0Hct5a7YG1syWZUg+FJ22\nH8N7qLOBjw5RcKCoepuRvMgP71+Hp03Xt8WSpN1Evl6EllwtmTtVTTeVS8fjP7xL\nhc7XemtDPY/81cBuj+HCit9/+44HZCT9V3dV6D9IWWnc3mECQQD1sTvcNAsh8idv\nMS12jmqdaOYTnJM1kFiddRvdkfChADq35x5bzV/oORYAmfurjuPN7ssHvrEEjmew\nbvi62MYtAkEAyDsAKrWsAfJQKbraTraJE7r7mTWxvAAYUONKKPZV2BXPzrTD/WMI\nn7z95pUu8x7anck9qqF6RYplo4fFLQKh/QJBANYwsszgGix33WUUbFwFAHFGN/40\n7CkwM/DhXW+mgS768jXNKSxDOS9MRSA1HbCMm5C2cw3Hcq9ULpUjyXeq7+kCQDx1\nvFYpJzgrP9Np7XNpILkJc+FOWk2nRbBfAUyfHUqzQ11qLef8GGWLfqs6jsOwpFiS\npIE6Yx5ObORVIc+2hM0CQE/pVhPEZ3boB8xoc9+3YL+++0yR2uMHoTY/q6r96kPC\n6C1oSRcDX/MUDOzC5HCUuwTYhNoN3FYkB5fov32BUbQ=\n-----END RSA PRIVATE KEY-----\n",
            "client_email": "none",
            "client_id": "114469363779822440226",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/vertex%40tensorzero-public.iam.gserviceaccount.com",
            "universe_domain": "googleapis.com"
        }"#;
        let generic = Credential::FileContents(SecretString::from(json_content));
        let creds = GCPVertexCredentials::try_from((generic, "GCPVertexGemini")).unwrap();
        assert!(matches!(creds, GCPVertexCredentials::Static { .. }));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = GCPVertexCredentials::try_from((generic, "GCPVertexGemini")).unwrap();
        assert!(matches!(creds, GCPVertexCredentials::Dynamic(_)));

        // Test Missing credential
        let generic = Credential::Missing;
        let creds = GCPVertexCredentials::try_from((generic, "GCPVertexGemini")).unwrap();
        assert!(matches!(creds, GCPVertexCredentials::None));

        // Test invalid JSON content
        let invalid_json = "invalid json";
        let generic = Credential::FileContents(SecretString::from(invalid_json));
        let result = GCPVertexCredentials::try_from((generic, "GCPVertexGemini"));
        assert!(result.is_err());
        let err = result.unwrap_err().get_owned_details();
        assert!(
            matches!(err, ErrorDetails::GCPCredentials { message } if message.contains("Failed to load GCP credentials"))
        );

        // Test invalid credential type (Static)
        let generic = Credential::Static(SecretString::from("test"));
        let result = GCPVertexCredentials::try_from((generic, "GCPVertexGemini"));
        assert!(result.is_err());
        let err = result.unwrap_err().get_owned_details();
        assert!(
            matches!(err, ErrorDetails::GCPCredentials { message } if message.contains("Invalid credential_location"))
        );
    }

    #[test]
    fn test_shorthand_url_parse() {
        use super::parse_shorthand_url;

        let err1 = parse_shorthand_url("bad-shorthand-url", "google")
            .unwrap_err()
            .to_string();
        assert_eq!(err1, "GCP shorthand url is not in the expected format (should start with `projects/<project_id>/locations/<location>'): `bad-shorthand-url`");

        let missing_components = parse_shorthand_url(
            "projects/tensorzero-public/locations/us-central1/",
            "google",
        )
        .unwrap_err()
        .to_string();
        assert_eq!(missing_components, "GCP shorthand url does not contain a publisher or endpoint: `projects/tensorzero-public/locations/us-central1/`");

        let non_google_publisher = parse_shorthand_url("projects/tensorzero-public/locations/us-central1/publishers/not-google/models/gemini-2.0-flash-001", "google").unwrap_err().to_string();
        assert_eq!(non_google_publisher, "GCP shorthand url has publisher `not-google`, expected `google` : `projects/tensorzero-public/locations/us-central1/publishers/not-google/models/gemini-2.0-flash-001`");

        let valid_model_url = parse_shorthand_url("projects/tensorzero-public/locations/us-central1/publishers/google/models/gemini-2.0-flash-001", "google").unwrap();
        assert_eq!(
            valid_model_url,
            ShorthandUrl::Publisher {
                location: "us-central1",
                model_id: "gemini-2.0-flash-001"
            }
        );

        let valid_endpoint_url = parse_shorthand_url(
            "projects/tensorzero-public/locations/us-central1/endpoints/945488740422254592",
            "google",
        )
        .unwrap();
        assert_eq!(
            valid_endpoint_url,
            ShorthandUrl::Endpoint {
                location: "us-central1",
                endpoint_id: "945488740422254592"
            }
        );
    }
}
