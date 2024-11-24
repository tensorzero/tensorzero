use axum::body::Body;
use axum::extract::State;
use axum::response::Response;
use axum::{debug_handler, Json};
use metrics::counter;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::Instant;
use tracing::instrument;
use uuid::Uuid;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::Config;
use crate::embeddings::EmbeddingModelConfig;
use crate::error::{Error, ErrorDetails};
use crate::function::sample_variant;
use crate::function::FunctionConfig;
use crate::gateway_util::{AppState, AppStateData, StructuredJson};
use crate::inference::types::{
    ChatInferenceDatabaseInsert, ContentBlockOutput, InferenceResult, Input,
    JsonInferenceDatabaseInsert, JsonInferenceOutput, ModelInferenceResponseWithMetadata,
    RequestMessage, Usage,
};
use crate::jsonschema_util::DynamicJSONSchema;
use crate::model::ModelConfig;
use crate::tool::{
    BatchDynamicToolParams, BatchDynamicToolParamsWithSize, DynamicToolParams, ToolCallConfig,
};
use crate::uuid_util::validate_episode_id;
use crate::variant::{BatchInferenceConfig, OwnedInferenceConfig, Variant};

use super::inference::{
    ChatCompletionInferenceParams, InferenceClients, InferenceModels, InferenceParams,
};

/// The expected payload is a JSON object with the following fields:
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Params {
    // the function name
    pub function_name: String,
    // the episode IDs for each inference (if not provided, it'll be set to inference_id)
    // NOTE: DO NOT GENERATE EPISODE IDS MANUALLY. THE API WILL DO THAT FOR YOU.
    #[serde(default)]
    pub episode_ids: Option<BatchEpisodeIdInput>,
    // the inputs for the inferences
    pub inputs: Vec<Input>,
    // Inference-time overrides for variant types (use with caution)
    #[serde(default)]
    pub params: BatchInferenceParams,
    // if the client would like to pin a specific variant to be used
    // NOTE: YOU SHOULD TYPICALLY LET THE API SELECT A VARIANT FOR YOU (I.E. IGNORE THIS FIELD).
    //       ONLY PIN A VARIANT FOR SPECIAL USE CASES (E.G. TESTING / DEBUGGING VARIANTS).
    pub variant_name: Option<String>,
    // the tags to add to the inference
    #[serde(default)]
    pub tags: Option<BatchTags>,
    // dynamic information about tool calling. Don't directly include `dynamic_tool_params` in `Params`.
    #[serde(flatten)]
    pub dynamic_tool_params: BatchDynamicToolParams,
    // `dynamic_tool_params` includes the following fields, passed at the top level of `Params`:
    // If provided, the inference will only use the specified tools (a subset of the function's tools)
    // allowed_tools: Option<Vec<Option<Vec<String>>>>,
    // If provided, the inference will use the specified tools in addition to the function's tools
    // additional_tools: Option<Vec<Option<Vec<Tool>>>>,
    // If provided, the inference will use the specified tool choice
    // tool_choice: Option<Vec<Option<ToolChoice>>>,
    // If true, the inference will use parallel tool calls
    // parallel_tool_calls: Option<Vec<Option<bool>>>,
    // If provided for a JSON inference, the inference will use the specified output schema instead of the
    // configured one. We only lazily validate this schema.
    #[serde(default)]
    pub output_schemas: Option<BatchOutputSchemas>,
    #[serde(default)]
    pub credentials: InferenceCredentials,
}

type BatchEpisodeIdInput = Vec<Option<Uuid>>;
type BatchEpisodeIds = Vec<Uuid>;
type BatchTags = Vec<Option<HashMap<String, String>>>;
type BatchOutputSchemas = Vec<Option<Value>>;

#[derive(Clone, Debug)]
struct InferenceMetadata<'a> {
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub input: Input,
    pub dryrun: bool,
    pub start_time: Instant,
    pub inference_params: BatchInferenceParams,
    pub model_name: &'a str,
    pub model_provider_name: &'a str,
    pub raw_request: String,
    pub system: Option<String>,
    pub input_messages: Vec<RequestMessage>,
    pub previous_model_inference_results: Vec<ModelInferenceResponseWithMetadata<'a>>,
    pub tags: HashMap<String, String>,
}

pub type InferenceCredentials = HashMap<String, SecretString>;

/// A handler for the inference endpoint
#[instrument(
    name="post_batch_inference",
    skip(config, http_client, clickhouse_connection_info, params),
    fields(
        function_name = %params.function_name,
        variant_name = ?params.variant_name,
    )
)]
#[debug_handler(state = AppStateData)]
pub async fn prepare_batch_inference_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
    }): AppState,
    StructuredJson(params): StructuredJson<Params>,
) -> Result<Response<Body>, Error> {
    // To be used for the Inference table processing_time measurements
    let start_time = Instant::now();
    // Get the function config or return an error if it doesn't exist
    let function = config.get_function(&params.function_name)?;
    let num_inferences = params.inputs.len();
    if num_inferences == 0 {
        return Err(ErrorDetails::InvalidRequest {
            message: "No inputs provided".to_string(),
        }
        .into());
    }
    let batch_dynamic_tool_params: Vec<DynamicToolParams> =
        BatchDynamicToolParamsWithSize(params.dynamic_tool_params, num_inferences).try_into()?;

    let tool_configs = batch_dynamic_tool_params
        .into_iter()
        .map(|dynamic_tool_params| function.prepare_tool_config(dynamic_tool_params, &config.tools))
        .collect::<Result<Vec<_>, _>>()?;
    // Collect the function variant names as a Vec<&str>
    let mut candidate_variant_names: Vec<&str> =
        function.variants().keys().map(AsRef::as_ref).collect();

    // If the function has no variants, return an error
    if candidate_variant_names.is_empty() {
        return Err(ErrorDetails::InvalidFunctionVariants {
            message: format!("Function `{}` has no variants", params.function_name),
        }
        .into());
    }

    // Validate the input
    params
        .inputs
        .iter()
        .enumerate()
        .try_for_each(|(i, input)| {
            function.validate_input(input).map_err(|e| {
                ErrorDetails::BatchInputValidation {
                    index: i,
                    message: e.to_string(),
                }
                .into()
            })
        })?;

    // If a variant is pinned, only that variant should be attempted
    if let Some(ref variant_name) = params.variant_name {
        candidate_variant_names.retain(|k| k == variant_name);

        // If the pinned variant doesn't exist, return an error
        if candidate_variant_names.is_empty() {
            return Err(ErrorDetails::UnknownVariant {
                name: variant_name.to_string(),
            }
            .into());
        }
    }

    // Retrieve or generate the episode IDs and validate them (in the impl)
    let episode_ids: BatchEpisodeIds =
        BatchEpisodeIdsWithSize(params.episode_ids, num_inferences).try_into()?;

    // Increment the request count if we're not in dryrun mode
    counter!(
        "request_count",
        "endpoint" => "post_batch_inference",
        "function_name" => params.function_name.to_string(),
    )
    .increment(1);
    counter!(
        "inference_count",
        "endpoint" => "post_batch_inference",
        "function_name" => params.function_name.to_string(),
    )
    .increment(num_inferences as u64);

    // Keep track of which variants failed
    let mut variant_errors = std::collections::HashMap::new();
    let mut inference_config = BatchInferenceConfig::new(
        params.function_name.clone(),
        &config.templates,
        tool_configs,
        params.output_schemas,
    );

    let inference_clients = InferenceClients {
        http_client: &http_client,
        clickhouse_connection_info: &clickhouse_connection_info,
        credentials: &params.credentials,
    };

    let inference_models = InferenceModels {
        models: &config.models,
        embedding_models: &config.embedding_models,
    };
    let inference_params: Vec<InferenceParams> =
        BatchInferenceParamsWithSize(params.params, num_inferences).try_into()?;

    // Keep sampling variants until one succeeds
    // We already guarantee there is at least one inference
    let first_episode_id = episode_ids
        .get(0)
        .ok_or_else(|| Error::new(ErrorDetails::Inference {
            message: "batch episode_ids unexpectedly empty. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
        }))?;
    while !candidate_variant_names.is_empty() {
        // We sample the same variant for the whole batch
        let (variant_name, variant) = sample_variant(
            &mut candidate_variant_names,
            function.variants(),
            &params.function_name,
            first_episode_id,
        )?;
        // Will be edited by the variant as part of making the request so we must clone here
        let variant_inference_params = inference_params.clone();
        inference_config.variant_name = variant_name.to_string();

        let result = variant
            .prepare_batch_inference(
                &params.inputs,
                &inference_models,
                function,
                &inference_config,
                &inference_clients,
                variant_inference_params,
            )
            .await;
        // TODO(Viraj): stopped here

        let result = match result {
            Ok(result) => result,
            Err(e) => {
                tracing::warn!(
                        "functions.{function_name}.variants.{variant_name} failed during inference: {e}",
                        function_name = params.function_name,
                        variant_name = variant_name,
                    );
                variant_errors.insert(variant_name.to_string(), e);
                continue;
            }
        };

        if !dryrun {
            // Spawn a thread for a trailing write to ClickHouse so that it doesn't block the response
            let write_metadata = InferenceDatabaseInsertMetadata {
                function_name: params.function_name,
                variant_name: variant_name.to_string(),
                episode_id,
                tool_params: inference_config.tool_config,
                processing_time: start_time.elapsed(),
                tags: params.tags,
            };

            let result_to_write = result.clone();

            tokio::spawn(async move {
                write_inference(
                    &clickhouse_connection_info,
                    params.input,
                    result_to_write,
                    write_metadata,
                )
                .await;
            });
        }

        let response = InferenceResponse::new(result, episode_id, variant_name.to_string());

        return Ok(InferenceOutput::NonStreaming(response));
    }

    // Eventually, if we get here, it means we tried every variant and none of them worked
    Err(ErrorDetails::AllVariantsFailed {
        errors: variant_errors,
    }
    .into())
}

#[derive(Debug)]
pub struct InferenceDatabaseInsertMetadata {
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub tool_params: Option<ToolCallConfig>,
    pub processing_time: Duration,
    pub tags: HashMap<String, String>,
}

async fn write_inference<'a>(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    input: Input,
    result: InferenceResult<'a>,
    metadata: InferenceDatabaseInsertMetadata,
) {
    let serialized_input = match serde_json::to_string(&input).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: e.to_string(),
        })
    }) {
        Ok(serialized_input) => serialized_input,
        Err(_) => {
            return;
        }
    };
    let model_responses: Vec<serde_json::Value> = result.get_serialized_model_inferences();
    // Write the model responses to the ModelInference table
    for response in model_responses {
        let _ = clickhouse_connection_info
            .write(&response, "ModelInference")
            .await;
    }
    // Write the inference to the Inference table
    match result {
        InferenceResult::Chat(result) => {
            let chat_inference =
                ChatInferenceDatabaseInsert::new(result, serialized_input, metadata);
            let _ = clickhouse_connection_info
                .write(&chat_inference, "ChatInference")
                .await;
        }
        InferenceResult::Json(result) => {
            let json_inference =
                JsonInferenceDatabaseInsert::new(result, serialized_input, metadata);
            let _ = clickhouse_connection_info
                .write(&json_inference, "JsonInference")
                .await;
        }
    }
}

/// InferenceResponse and InferenceResultChunk determine what gets serialized and sent to the client

#[derive(Clone, Debug, Serialize)]
#[serde(untagged, rename_all = "snake_case")]
pub enum InferenceResponse {
    Chat(ChatInferenceResponse),
    Json(JsonInferenceResponse),
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatInferenceResponse {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub content: Vec<ContentBlockOutput>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize)]
pub struct JsonInferenceResponse {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub output: JsonInferenceOutput,
    pub usage: Usage,
}

impl InferenceResponse {
    fn new(inference_result: InferenceResult, episode_id: Uuid, variant_name: String) -> Self {
        match inference_result {
            InferenceResult::Chat(result) => InferenceResponse::Chat(ChatInferenceResponse {
                inference_id: result.inference_id,
                episode_id,
                variant_name,
                content: result.content,
                usage: result.usage,
            }),
            InferenceResult::Json(result) => InferenceResponse::Json(JsonInferenceResponse {
                inference_id: result.inference_id,
                episode_id,
                variant_name,
                output: result.output,
                usage: result.usage,
            }),
        }
    }
}

struct BatchEpisodeIdsWithSize(Option<BatchEpisodeIdInput>, usize);

impl TryFrom<BatchEpisodeIdsWithSize> for BatchEpisodeIds {
    type Error = Error;

    fn try_from(
        BatchEpisodeIdsWithSize(episode_ids, num_inferences): BatchEpisodeIdsWithSize,
    ) -> Result<Self, Self::Error> {
        let episode_ids = match episode_ids {
            Some(episode_ids) => {
                if episode_ids.len() != num_inferences {
                    return Err(ErrorDetails::InvalidRequest {
                        message: format!(
                            "Number of episode_ids ({}) does not match number of inputs ({})",
                            episode_ids.len(),
                            num_inferences
                        ),
                    }
                    .into());
                }

                episode_ids
                    .into_iter()
                    .map(|id| id.unwrap_or_else(Uuid::now_v7))
                    .collect()
            }
            None => vec![Uuid::now_v7(); num_inferences],
        };
        episode_ids.iter().enumerate().try_for_each(|(i, id)| {
            validate_episode_id(*id).map_err(|e| {
                Error::new(ErrorDetails::BatchInputValidation {
                    index: i,
                    message: e.to_string(),
                })
            })
        })?;
        Ok(episode_ids)
    }
}

/// InferenceParams is the top-level struct for inference parameters.
/// We backfill these from the configs given in the variants used and ultimately write them to the database.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
struct BatchInferenceParams {
    pub chat_completion: BatchChatCompletionInferenceParams,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
struct BatchChatCompletionInferenceParams {
    #[serde(default)]
    pub temperature: Option<Vec<Option<f32>>>,
    #[serde(default)]
    pub max_tokens: Option<Vec<Option<u32>>>,
    #[serde(default)]
    pub seed: Option<Vec<Option<u32>>>,
    #[serde(default)]
    pub top_p: Option<Vec<Option<f32>>>,
    #[serde(default)]
    pub presence_penalty: Option<Vec<Option<f32>>>,
    #[serde(default)]
    pub frequency_penalty: Option<Vec<Option<f32>>>,
}

struct BatchInferenceParamsWithSize(BatchInferenceParams, usize);
impl TryFrom<BatchInferenceParamsWithSize> for Vec<InferenceParams> {
    type Error = Error;

    fn try_from(
        BatchInferenceParamsWithSize(params, num_inferences): BatchInferenceParamsWithSize,
    ) -> Result<Self, Self::Error> {
        let BatchInferenceParams { chat_completion } = params;
        let chat_completion_params: Vec<ChatCompletionInferenceParams> =
            BatchChatCompletionParamsWithSize(chat_completion, num_inferences).try_into()?;
        Ok(chat_completion_params
            .into_iter()
            .map(|p| InferenceParams { chat_completion: p })
            .collect())
    }
}

struct BatchChatCompletionParamsWithSize(BatchChatCompletionInferenceParams, usize);
impl TryFrom<BatchChatCompletionParamsWithSize> for Vec<ChatCompletionInferenceParams> {
    type Error = Error;

    fn try_from(
        BatchChatCompletionParamsWithSize(params, num_inferences): BatchChatCompletionParamsWithSize,
    ) -> Result<Self, Self::Error> {
        let BatchChatCompletionInferenceParams {
            temperature,
            max_tokens,
            seed,
            top_p,
            presence_penalty,
            frequency_penalty,
        } = params;
        // Verify all provided Vecs have the same length
        if let Some(temperature) = &temperature {
            if temperature.len() != num_inferences {
                return Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "temperature vector length ({}) does not match number of inferences ({})",
                        temperature.len(),
                        num_inferences
                    ),
                }
                .into());
            }
        }

        if let Some(max_tokens) = &max_tokens {
            if max_tokens.len() != num_inferences {
                return Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "max_tokens vector length ({}) does not match number of inferences ({})",
                        max_tokens.len(),
                        num_inferences
                    ),
                }
                .into());
            }
        }

        if let Some(seed) = &seed {
            if seed.len() != num_inferences {
                return Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "seed vector length ({}) does not match number of inferences ({})",
                        seed.len(),
                        num_inferences
                    ),
                }
                .into());
            }
        }

        if let Some(top_p) = &top_p {
            if top_p.len() != num_inferences {
                return Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "top_p vector length ({}) does not match number of inferences ({})",
                        top_p.len(),
                        num_inferences
                    ),
                }
                .into());
            }
        }

        if let Some(presence_penalty) = &presence_penalty {
            if presence_penalty.len() != num_inferences {
                return Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "presence_penalty vector length ({}) does not match number of inferences ({})",
                        presence_penalty.len(),
                        num_inferences
                    ),
                }
                .into());
            }
        }

        if let Some(frequency_penalty) = &frequency_penalty {
            if frequency_penalty.len() != num_inferences {
                return Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "frequency_penalty vector length ({}) does not match number of inferences ({})",
                        frequency_penalty.len(),
                        num_inferences
                    ),
                }
                .into());
            }
        }

        // Convert Option<Vec<Option<T>>> into Vec<Option<T>> by unwrapping or creating empty vec
        let temperature = temperature.unwrap_or_default();
        let max_tokens = max_tokens.unwrap_or_default();
        let seed = seed.unwrap_or_default();
        let top_p = top_p.unwrap_or_default();
        let presence_penalty = presence_penalty.unwrap_or_default();
        let frequency_penalty = frequency_penalty.unwrap_or_default();

        // Create iterators that take ownership
        let mut temperature_iter = temperature.into_iter();
        let mut max_tokens_iter = max_tokens.into_iter();
        let mut seed_iter = seed.into_iter();
        let mut top_p_iter = top_p.into_iter();
        let mut presence_penalty_iter = presence_penalty.into_iter();
        let mut frequency_penalty_iter = frequency_penalty.into_iter();

        // Build params using the iterators
        let mut all_inference_params = Vec::with_capacity(num_inferences);
        for _ in 0..num_inferences {
            all_inference_params.push(ChatCompletionInferenceParams {
                temperature: temperature_iter.next().unwrap_or(None),
                max_tokens: max_tokens_iter.next().unwrap_or(None),
                seed: seed_iter.next().unwrap_or(None),
                top_p: top_p_iter.next().unwrap_or(None),
                presence_penalty: presence_penalty_iter.next().unwrap_or(None),
                frequency_penalty: frequency_penalty_iter.next().unwrap_or(None),
            });
        }
        Ok(all_inference_params)
    }
}
