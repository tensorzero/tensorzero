use crate::{
    endpoints::{
        batch_inference::{BatchEpisodeIdInput, BatchOutputSchemas},
        inference::{ChatCompletionInferenceParams, InferenceParams},
    },
    error::{Error, ErrorDetails},
    jsonschema_util::DynamicJSONSchema,
    tool::{ToolCallConfig, ToolCallConfigDatabaseInsert},
    uuid_util::validate_episode_id,
};

use super::{Input, ModelInferenceRequest, RequestMessage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{borrow::Cow, collections::HashMap};
use uuid::Uuid;

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    Pending,
    Completed,
    Failed,
}

/// Returned from start_batch_inference from an InferenceProvider
/// This is the original response type for start_batch_inference.
/// The types below add additional context to the response as it is returned
/// through the call stack.
pub struct BatchProviderInferenceResponse {
    pub batch_id: Uuid,
    pub inference_ids: Vec<Uuid>,
    pub raw_requests: Vec<String>,
    pub batch_params: Value,
    pub status: BatchStatus,
}

/// Returned from start_batch_inference from a model
/// Adds the model provider name to the response
pub struct BatchModelInferenceResponse<'a> {
    pub batch_id: Uuid,
    pub inference_ids: Vec<Uuid>,
    pub raw_requests: Vec<String>,
    pub batch_params: Value,
    pub model_provider_name: &'a str,
    pub status: BatchStatus,
}

impl<'a> BatchModelInferenceResponse<'a> {
    pub fn new(
        provider_batch_response: BatchProviderInferenceResponse,
        model_provider_name: &'a str,
    ) -> Self {
        Self {
            batch_id: provider_batch_response.batch_id,
            inference_ids: provider_batch_response.inference_ids,
            raw_requests: provider_batch_response.raw_requests,
            batch_params: provider_batch_response.batch_params,
            model_provider_name,
            status: provider_batch_response.status,
        }
    }
}

/// Returned from poll_batch_inference from a variant.
/// Here, we add context from the variant, such as the original inputs, templated input messages,
/// systems, tool configs, inference params, model_name, and output schemas.
pub struct BatchModelInferenceWithMetadata<'a> {
    pub batch_id: Uuid,
    pub inference_ids: Vec<Uuid>,
    pub input_messages: Vec<Vec<RequestMessage>>,
    pub systems: Vec<Option<String>>,
    pub tool_configs: Vec<Option<Cow<'a, ToolCallConfig>>>,
    pub inference_params: Vec<InferenceParams>,
    pub output_schemas: Vec<Option<&'a Value>>,
    pub raw_requests: Vec<String>,
    pub batch_params: Value,
    pub model_provider_name: &'a str,
    pub model_name: &'a str,
    pub status: BatchStatus,
}

impl<'a> BatchModelInferenceWithMetadata<'a> {
    pub fn new(
        model_batch_response: BatchModelInferenceResponse<'a>,
        model_inference_requests: Vec<ModelInferenceRequest<'a>>,
        model_name: &'a str,
        inference_params: Vec<InferenceParams>,
    ) -> Self {
        let mut input_messages: Vec<Vec<RequestMessage>> = vec![];
        let mut systems: Vec<Option<String>> = vec![];
        let mut tool_configs: Vec<Option<Cow<'a, ToolCallConfig>>> = vec![];
        let mut output_schemas: Vec<Option<&'a Value>> = vec![];
        for request in model_inference_requests {
            input_messages.push(request.messages);
            systems.push(request.system);
            tool_configs.push(request.tool_config);
            output_schemas.push(request.output_schema);
        }
        Self {
            batch_id: model_batch_response.batch_id,
            inference_ids: model_batch_response.inference_ids,
            input_messages,
            systems,
            tool_configs,
            inference_params,
            output_schemas,
            raw_requests: model_batch_response.raw_requests,
            batch_params: model_batch_response.batch_params,
            model_provider_name: model_batch_response.model_provider_name,
            model_name,
            status: model_batch_response.status,
        }
    }
}

/// Additional metadata needed to write to ClickHouse that isn't available at the variant level
#[derive(Debug)]
pub struct BatchInferenceDatabaseInsertMetadata<'a> {
    pub function_name: &'a str,
    pub variant_name: &'a str,
    pub episode_ids: &'a Vec<Uuid>,
    pub tags: Option<Vec<Option<HashMap<String, String>>>>,
}

/// Data needed to write to the `BatchModelInference` table in ClickHouse
///
/// Design constraint: this should contain all the information needed from
/// starting batch inference to eventually populate ChatInference, JsonInference, and ModelInference
/// tables.
#[derive(Debug, Serialize)]
pub struct BatchModelInferenceRow<'a> {
    pub inference_id: Uuid,
    pub batch_id: &'a str,
    pub function_name: &'a str,
    pub variant_name: &'a str,
    pub episode_id: Uuid,
    pub input: Input,
    pub input_messages: Vec<RequestMessage>,
    pub system: Option<&'a str>,
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    pub inference_params: &'a InferenceParams,
    pub output_schema: Option<&'a Value>,
    pub raw_request: &'a str,
    pub model_name: &'a str,
    pub model_provider_name: &'a str,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct BatchRequestRow<'a> {
    batch_id: &'a str,
    id: String,
    batch_params: String,
    model_name: &'a str,
    model_provider_name: &'a str,
    status: BatchStatus,
    errors: HashMap<String, String>,
}

impl<'a> BatchRequestRow<'a> {
    pub fn new(
        batch_id: &'a str,
        batch_params: Value,
        model_name: &'a str,
        model_provider_name: &'a str,
        status: BatchStatus,
        errors: Option<HashMap<String, String>>,
    ) -> Self {
        let id = Uuid::now_v7().to_string();
        let errors = errors.unwrap_or_default();
        Self {
            batch_id,
            id,
            batch_params: serde_json::to_string(&batch_params)
                .map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: e.to_string(),
                    })
                })
                .unwrap_or_default(),
            model_name,
            model_provider_name,
            status,
            errors,
        }
    }
}

/// Below are types required for parsing and processing inputs for batch inference requests.
/// The idea here is that we need to get a vector of the length of the number of inferences
/// so that we can use existing code.

pub struct BatchEpisodeIdsWithSize(pub Option<BatchEpisodeIdInput>, pub usize);
pub type BatchEpisodeIds = Vec<Uuid>;

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
pub struct BatchInferenceParams {
    pub chat_completion: BatchChatCompletionInferenceParams,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct BatchChatCompletionInferenceParams {
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

pub struct BatchInferenceParamsWithSize(pub BatchInferenceParams, pub usize);

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

pub struct BatchChatCompletionParamsWithSize(BatchChatCompletionInferenceParams, usize);

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

pub struct BatchOutputSchemasWithSize(pub Option<BatchOutputSchemas>, pub usize);

impl TryFrom<BatchOutputSchemasWithSize> for Vec<Option<DynamicJSONSchema>> {
    type Error = Error;

    fn try_from(
        BatchOutputSchemasWithSize(schemas, num_inferences): BatchOutputSchemasWithSize,
    ) -> Result<Self, Self::Error> {
        if let Some(schemas) = schemas {
            if schemas.len() != num_inferences {
                Err(ErrorDetails::InvalidRequest {
                    message: format!(
                        "output_schemas vector length ({}) does not match number of inferences ({})",
                        schemas.len(),
                        num_inferences
                    ),
                }
                .into())
            } else {
                Ok(schemas
                    .into_iter()
                    .map(|schema| schema.map(DynamicJSONSchema::new))
                    .collect())
            }
        } else {
            Ok(vec![None; num_inferences])
        }
    }
}

#[cfg(test)]
mod tests {
    use uuid::Timestamp;

    use super::*;

    #[test]
    fn test_try_from_batch_episode_ids_with_size() {
        let batch_episode_ids_with_size = BatchEpisodeIdsWithSize(None, 3);
        let batch_episode_ids = BatchEpisodeIds::try_from(batch_episode_ids_with_size).unwrap();
        assert_eq!(batch_episode_ids.len(), 3);

        let batch_episode_ids_with_size = BatchEpisodeIdsWithSize(Some(vec![None, None, None]), 3);
        let batch_episode_ids = BatchEpisodeIds::try_from(batch_episode_ids_with_size).unwrap();
        assert_eq!(batch_episode_ids.len(), 3);

        let episode_id_0 = Uuid::now_v7();
        let episode_id_1 = Uuid::now_v7();
        let batch_episode_ids_with_size =
            BatchEpisodeIdsWithSize(Some(vec![Some(episode_id_0), Some(episode_id_1), None]), 3);
        let batch_episode_ids = BatchEpisodeIds::try_from(batch_episode_ids_with_size).unwrap();
        assert_eq!(batch_episode_ids.len(), 3);
        assert_eq!(batch_episode_ids[0], episode_id_0);
        assert_eq!(batch_episode_ids[1], episode_id_1);

        let early_uuid = Uuid::new_v7(Timestamp::from_unix_time(946766218, 0, 0, 0));
        let batch_episode_ids_with_size =
            BatchEpisodeIdsWithSize(Some(vec![Some(early_uuid), None, None]), 3);
        let err = BatchEpisodeIds::try_from(batch_episode_ids_with_size).unwrap_err();
        assert_eq!(
            err,
            ErrorDetails::BatchInputValidation {
                index: 0,
                message: "Invalid Episode ID: Timestamp is too early".to_string(),
            }
            .into()
        );
    }

    #[test]
    fn test_batch_inference_params_with_size() {
        // Try with default params
        let batch_inference_params_with_size =
            BatchInferenceParamsWithSize(BatchInferenceParams::default(), 3);
        let inference_params =
            Vec::<InferenceParams>::try_from(batch_inference_params_with_size).unwrap();
        assert_eq!(inference_params.len(), 3);
        assert_eq!(
            inference_params[0].chat_completion,
            ChatCompletionInferenceParams::default()
        );

        // Try with some overridden params
        let batch_inference_params_with_size = BatchInferenceParamsWithSize(
            BatchInferenceParams {
                chat_completion: BatchChatCompletionInferenceParams {
                    temperature: Some(vec![Some(0.5), None, None]),
                    max_tokens: Some(vec![None, None, Some(30)]),
                    seed: Some(vec![None, Some(2), Some(3)]),
                    top_p: None,
                    presence_penalty: Some(vec![Some(0.5), Some(0.6), Some(0.7)]),
                    frequency_penalty: Some(vec![Some(0.5), Some(0.6), Some(0.7)]),
                },
            },
            3,
        );

        let inference_params =
            Vec::<InferenceParams>::try_from(batch_inference_params_with_size).unwrap();
        assert_eq!(inference_params.len(), 3);
        assert_eq!(inference_params[0].chat_completion.temperature, Some(0.5));
        assert_eq!(inference_params[1].chat_completion.max_tokens, None);
        assert_eq!(inference_params[2].chat_completion.seed, Some(3));
        // Check top_p is None for all since it wasn't specified
        assert_eq!(inference_params[0].chat_completion.top_p, None);
        assert_eq!(inference_params[1].chat_completion.top_p, None);
        assert_eq!(inference_params[2].chat_completion.top_p, None);

        // Check presence_penalty values
        assert_eq!(
            inference_params[0].chat_completion.presence_penalty,
            Some(0.5)
        );
        assert_eq!(
            inference_params[1].chat_completion.presence_penalty,
            Some(0.6)
        );
        assert_eq!(
            inference_params[2].chat_completion.presence_penalty,
            Some(0.7)
        );

        // Check frequency_penalty values
        assert_eq!(
            inference_params[0].chat_completion.frequency_penalty,
            Some(0.5)
        );
        assert_eq!(
            inference_params[1].chat_completion.frequency_penalty,
            Some(0.6)
        );
        assert_eq!(
            inference_params[2].chat_completion.frequency_penalty,
            Some(0.7)
        );

        // Verify temperature is None for indices 1 and 2
        assert_eq!(inference_params[1].chat_completion.temperature, None);
        assert_eq!(inference_params[2].chat_completion.temperature, None);

        // Verify max_tokens is 30 for last item and None for first
        assert_eq!(inference_params[0].chat_completion.max_tokens, None);
        assert_eq!(inference_params[2].chat_completion.max_tokens, Some(30));

        // Verify seed is None for first item and 2 for second
        assert_eq!(inference_params[0].chat_completion.seed, None);
        assert_eq!(inference_params[1].chat_completion.seed, Some(2));

        // Test with ragged arrays (arrays of different lengths)
        let batch_inference_params_with_size = BatchInferenceParamsWithSize(
            BatchInferenceParams {
                chat_completion: BatchChatCompletionInferenceParams {
                    temperature: Some(vec![Some(0.5), None]), // Too short
                    max_tokens: Some(vec![None, None, Some(30), Some(40)]), // Too long
                    seed: Some(vec![]),                       // Empty array
                    top_p: None,
                    presence_penalty: Some(vec![Some(0.5)]), // Too short
                    frequency_penalty: Some(vec![Some(0.5), Some(0.6), Some(0.7), Some(0.8)]), // Too long
                },
            },
            3,
        );

        let err = Vec::<InferenceParams>::try_from(batch_inference_params_with_size).unwrap_err();
        match err.get_details() {
            ErrorDetails::InvalidRequest { message } => assert_eq!(
                message,
                "temperature vector length (2) does not match number of inferences (3)"
            ),
            _ => panic!("Expected InvalidRequest error"),
        }

        // Test with wrong size specified
        let batch_inference_params_with_size = BatchInferenceParamsWithSize(
            BatchInferenceParams {
                chat_completion: BatchChatCompletionInferenceParams {
                    temperature: Some(vec![Some(0.5), None, None, None]),
                    max_tokens: Some(vec![None, None, Some(30)]),
                    seed: Some(vec![None, Some(2), Some(3)]),
                    top_p: None,
                    presence_penalty: Some(vec![Some(0.5), Some(0.6), Some(0.7)]),
                    frequency_penalty: Some(vec![Some(0.5), Some(0.6), Some(0.7)]),
                },
            },
            4, // Wrong size - arrays are length 3 but size is 4
        );

        let err = Vec::<InferenceParams>::try_from(batch_inference_params_with_size).unwrap_err();
        match err.get_details() {
            ErrorDetails::InvalidRequest { message } => assert_eq!(
                message,
                "max_tokens vector length (3) does not match number of inferences (4)"
            ),
            _ => panic!("Expected InvalidRequest error"),
        }
    }

    #[test]
    fn test_batch_output_schemas_with_size() {
        let batch_output_schemas_with_size = BatchOutputSchemasWithSize(None, 3);
        let batch_output_schemas =
            Vec::<Option<DynamicJSONSchema>>::try_from(batch_output_schemas_with_size).unwrap();
        assert_eq!(batch_output_schemas.len(), 3);

        let batch_output_schemas_with_size =
            BatchOutputSchemasWithSize(Some(vec![None, None, None]), 3);
        let batch_output_schemas =
            Vec::<Option<DynamicJSONSchema>>::try_from(batch_output_schemas_with_size).unwrap();
        assert_eq!(batch_output_schemas.len(), 3);
    }
}
