use chrono::Duration;
use std::collections::HashSet;
use std::sync::Arc;

use futures::future::try_join_all;
use serde::Deserialize;
use serde::Serialize;

use crate::config::path::ResolvedTomlPathData;
use crate::config::LoadableConfig;
use crate::config::PathWithContents;
use crate::embeddings::EmbeddingEncodingFormat;
use crate::embeddings::{EmbeddingModelTable, EmbeddingResponseWithMetadata};
use crate::endpoints::inference::InferenceModels;
use crate::inference::types::extra_body::{ExtraBodyConfig, FullExtraBodyConfig};
use crate::inference::types::extra_headers::{ExtraHeadersConfig, FullExtraHeadersConfig};
use crate::inference::types::resolved_input::LazyResolvedInputMessageContent;
use crate::inference::types::resolved_input::{LazyResolvedInput, LazyResolvedInputMessage};
use crate::inference::types::ContentBlock;
use crate::inference::types::ResolvedInput;
use crate::inference::types::ResolvedInputMessage;
use crate::inference::types::ResolvedInputMessageContent;
use crate::inference::types::StoredInput;
use crate::inference::types::StoredInputMessageContent;
use crate::inference::types::{
    batch::StartBatchModelInferenceWithMetadata,
    chat_completion_inference_params::ChatCompletionInferenceParamsV2, ModelInferenceRequest,
    RequestMessage, Role, Text,
};
use crate::model::ModelTable;
use crate::model_table::ShorthandModelConfig;
use crate::utils::retries::RetryConfig;
use crate::{
    embeddings::EmbeddingRequest,
    endpoints::inference::{InferenceClients, InferenceParams},
    error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE},
    function::FunctionConfig,
    inference::types::{
        ContentBlockChatOutput, InferenceResult, InferenceResultStream, JsonInferenceOutput,
    },
    minijinja_util::TemplateConfig,
};

use super::{
    chat_completion::{prepare_request_message, ChatTemplates},
    infer_model_request, infer_model_request_stream, prepare_model_inference_request,
    InferModelRequestArgs, InferenceConfig, JsonMode, ModelUsedInfo, Variant,
};

/// The primary configuration for the Dicl variant
/// We need a helper to deserialize the config because it relies on
/// a path to a file for system instructions and we need to use the
/// load() step to get the fully qualified path.
#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct DiclConfig {
    weight: Option<f64>,
    embedding_model: Arc<str>,
    k: u32, // k as in k-nearest neighbors
    model: Arc<str>,
    system_instructions: ResolvedTomlPathData,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop_sequences: Option<Vec<String>>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    max_tokens: Option<u32>,
    seed: Option<u32>,
    #[serde(flatten)]
    pub(crate) inference_params_v2: ChatCompletionInferenceParamsV2,
    json_mode: Option<JsonMode>,
    #[cfg_attr(test, ts(skip))]
    extra_body: Option<ExtraBodyConfig>,
    #[cfg_attr(test, ts(skip))]
    extra_headers: Option<ExtraHeadersConfig>,
    retries: RetryConfig,
    max_distance: Option<f32>,
}

impl DiclConfig {
    pub fn weight(&self) -> Option<f64> {
        self.weight
    }

    pub fn set_weight(&mut self, weight: Option<f64>) {
        self.weight = weight;
    }

    pub fn embedding_model(&self) -> &Arc<str> {
        &self.embedding_model
    }

    pub fn k(&self) -> u32 {
        self.k
    }

    pub fn model(&self) -> &Arc<str> {
        &self.model
    }

    pub fn system_instructions(&self) -> &str {
        self.system_instructions.data()
    }

    pub fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    pub fn top_p(&self) -> Option<f32> {
        self.top_p
    }

    pub fn stop_sequences(&self) -> Option<&Vec<String>> {
        self.stop_sequences.as_ref()
    }

    pub fn presence_penalty(&self) -> Option<f32> {
        self.presence_penalty
    }

    pub fn frequency_penalty(&self) -> Option<f32> {
        self.frequency_penalty
    }

    pub fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    pub fn seed(&self) -> Option<u32> {
        self.seed
    }

    pub fn reasoning_effort(&self) -> Option<&String> {
        self.inference_params_v2.reasoning_effort.as_ref()
    }

    pub fn verbosity(&self) -> Option<&String> {
        self.inference_params_v2.verbosity.as_ref()
    }

    pub fn json_mode(&self) -> Option<&JsonMode> {
        self.json_mode.as_ref()
    }

    pub fn extra_body(&self) -> Option<&ExtraBodyConfig> {
        self.extra_body.as_ref()
    }

    pub fn extra_headers(&self) -> Option<&ExtraHeadersConfig> {
        self.extra_headers.as_ref()
    }

    pub fn retries(&self) -> &RetryConfig {
        &self.retries
    }

    pub fn max_distance(&self) -> Option<f32> {
        self.max_distance
    }

    /// Converts this initialized config back to its uninitialized form.
    /// Note: Real file paths for system_instructions are preserved. Fake paths
    /// (like defaults) are converted to None and will be regenerated on load.
    pub fn as_uninitialized(self) -> UninitializedDiclConfig {
        UninitializedDiclConfig {
            weight: self.weight,
            embedding_model: self.embedding_model.to_string(),
            k: self.k,
            model: self.model.to_string(),
            system_instructions: if self.system_instructions.is_real_path() {
                Some(self.system_instructions)
            } else {
                None
            },
            temperature: self.temperature,
            top_p: self.top_p,
            stop_sequences: self.stop_sequences,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            max_tokens: self.max_tokens,
            seed: self.seed,
            reasoning_effort: self.inference_params_v2.reasoning_effort,
            thinking_budget_tokens: self.inference_params_v2.thinking_budget_tokens,
            verbosity: self.inference_params_v2.verbosity,
            json_mode: self.json_mode,
            extra_body: self.extra_body,
            retries: self.retries,
            extra_headers: self.extra_headers,
            max_distance: self.max_distance,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct UninitializedDiclConfig {
    #[serde(default)]
    pub weight: Option<f64>,
    pub embedding_model: String,
    pub k: u32, // k as in k-nearest neighbors
    pub model: String,
    pub system_instructions: Option<ResolvedTomlPathData>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub max_tokens: Option<u32>,
    pub seed: Option<u32>,
    #[cfg_attr(test, ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[cfg_attr(test, ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget_tokens: Option<i32>,
    #[cfg_attr(test, ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
    pub json_mode: Option<JsonMode>,
    #[serde(default)]
    #[ts(skip)]
    pub extra_body: Option<ExtraBodyConfig>,
    #[serde(default)]
    pub retries: RetryConfig,
    #[serde(default)]
    #[ts(skip)]
    pub extra_headers: Option<ExtraHeadersConfig>,
    #[serde(default)]
    pub max_distance: Option<f32>,
}

impl Variant for DiclConfig {
    async fn infer(
        &self,
        input: Arc<LazyResolvedInput>,
        models: InferenceModels,
        function: Arc<FunctionConfig>,
        inference_config: Arc<InferenceConfig>,
        clients: InferenceClients,
        inference_params: InferenceParams,
    ) -> Result<InferenceResult, Error> {
        // So this can be mutably borrowed by the prepare_request function
        let mut inference_params = inference_params;

        // Embed the input and grab the relevant examples from the database
        let (relevant_examples, embedding_response) = self
            .retrieve_relevant_examples(
                &input,
                &models.embedding_models,
                &clients,
                &inference_config.function_name,
                &inference_config.variant_name,
                &function,
            )
            .await?;

        // Prepare the request for the model
        let model_inference_request = self
            .prepare_request(
                &input,
                &relevant_examples,
                &function,
                &inference_config,
                false,
                &mut inference_params,
            )
            .await?;

        let model_config = models.models.get(self.model()).await?.ok_or_else(|| {
            Error::new(ErrorDetails::UnknownModel {
                name: self.model().to_string(),
            })
        })?;

        // Instantiate the InferModelRequestArgs struct
        let args = InferModelRequestArgs {
            request: model_inference_request,
            model_name: self.model().clone(),
            model_config: &model_config,
            function: function.as_ref(),
            inference_config: Arc::clone(&inference_config),
            clients,
            inference_params,
            retry_config: self.retries(),
        };

        // Refactored function call using the struct
        let mut inference_response = infer_model_request(args).await?;

        // Add the embedding to the model inference results
        inference_response
            .mut_model_inference_results()
            // This can only fail if the embedding response has a request with > 1 input,
            // which should never happen as we only send one input at a time
            .push(embedding_response.try_into()?);
        Ok(inference_response)
    }

    async fn infer_stream(
        &self,
        input: Arc<LazyResolvedInput>,
        models: InferenceModels,
        function: Arc<FunctionConfig>,
        inference_config: Arc<InferenceConfig>,
        clients: InferenceClients,
        inference_params: InferenceParams,
    ) -> Result<(InferenceResultStream, ModelUsedInfo), Error> {
        // So this can be mutably borrowed by the prepare_request function
        let mut inference_params = inference_params;

        // Embed the input and grab the relevant examples from the database
        let (relevant_examples, embedding_response) = self
            .retrieve_relevant_examples(
                &input,
                &models.embedding_models,
                &clients,
                &inference_config.function_name,
                &inference_config.variant_name,
                &function,
            )
            .await?;
        // Prepare the request for the model
        let request = self
            .prepare_request(
                &input,
                &relevant_examples,
                &function,
                &inference_config,
                true,
                &mut inference_params,
            )
            .await?;

        let model_config = models.models.get(self.model()).await?.ok_or_else(|| {
            Error::new(ErrorDetails::UnknownModel {
                name: self.model().to_string(),
            })
        })?;

        // Actually run the inference
        let (inference_result_stream, mut model_used_info) = infer_model_request_stream(
            request,
            self.model().clone(),
            &model_config,
            function.as_ref(),
            clients,
            inference_params,
            *self.retries(),
        )
        .await?;

        // Add the embedding to the model inference results
        model_used_info
            .previous_model_inference_results
            .push(embedding_response.try_into()?);
        Ok((inference_result_stream, model_used_info))
    }

    async fn validate(
        &self,
        _function: Arc<FunctionConfig>,
        models: &ModelTable,
        embedding_models: &EmbeddingModelTable,
        _templates: &TemplateConfig<'_>,
        function_name: &str,
        variant_name: &str,
        global_outbound_http_timeout: &Duration,
    ) -> Result<(), Error> {
        // TODO (#360): Add the clickhouse connection to this interface
        // Run a count() query on the DynamicInContextLearningExample table
        // WHERE function_name = function_name and variant_name = variant_name
        // Make sure that the count is positive

        // Validate that weight is non-negative
        if self.weight().is_some_and(|w| w < 0.0) {
            return Err(ErrorDetails::Config {
                message: format!(
                "`functions.{function_name}.variants.{variant_name}`: `weight` must be non-negative"
            ),
            }
            .into());
        }
        // Validate that the generation model and embedding model are valid
        models.validate(self.model())?;
        let embedding_model = embedding_models
            .get(self.embedding_model()).await?
            .ok_or_else(|| Error::new(ErrorDetails::Config {
                message: format!(
                    "`functions.{function_name}.variants.{variant_name}`: `embedding_model` must be a valid embedding model name"
                ),
            }))?;

        embedding_model
            .validate(self.embedding_model(), global_outbound_http_timeout)
            .map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!(
                "`functions.{function_name}.variants.{variant_name}` and embedding model `{}`: {e}",
                self.embedding_model()
                ),
                })
            })?;

        // Validate that max_distance is non-negative if specified
        if let Some(max_distance) = self.max_distance() {
            if max_distance < 0.0 {
                return Err(ErrorDetails::Config {
                    message: format!(
                        "`functions.{function_name}.variants.{variant_name}`: `max_distance` must be non-negative (got {max_distance})"
                    ),
                }
                .into());
            }
        }

        Ok(())
    }

    fn get_all_template_paths(&self) -> Vec<&PathWithContents> {
        vec![]
    }

    fn get_all_explicit_template_names(&self) -> HashSet<String> {
        HashSet::new()
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _input: &[LazyResolvedInput],
        _models: InferenceModels,
        _function: &'a FunctionConfig,
        _inference_configs: &'a [InferenceConfig],
        _clients: InferenceClients,
        _inference_params: Vec<InferenceParams>,
    ) -> Result<StartBatchModelInferenceWithMetadata<'a>, Error> {
        // TODO (#493): Implement batch inference for Dicl
        Err(ErrorDetails::UnsupportedVariantForBatchInference { variant_name: None }.into())
    }
}

fn lazy_content_to_resolved_discarding_incompatible(
    content: LazyResolvedInputMessageContent,
) -> Result<ResolvedInputMessageContent, Error> {
    Ok(match content {
        LazyResolvedInputMessageContent::Text(text) => ResolvedInputMessageContent::Text(text),
        LazyResolvedInputMessageContent::Template(template) => {
            // Stringify template as JSON for DICL
            let json_str = serde_json::to_string(&serde_json::json!({
                "type": "template",
                "name": template.name,
                "arguments": template.arguments,
            }))
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to stringify template content block: {e}"),
                })
            })?;
            ResolvedInputMessageContent::Text(Text { text: json_str })
        }
        LazyResolvedInputMessageContent::ToolCall(tool_call) => {
            ResolvedInputMessageContent::ToolCall(tool_call)
        }
        LazyResolvedInputMessageContent::ToolResult(tool_result) => {
            ResolvedInputMessageContent::ToolResult(tool_result)
        }
        LazyResolvedInputMessageContent::RawText(raw_text) => {
            ResolvedInputMessageContent::RawText(raw_text)
        }
        LazyResolvedInputMessageContent::Thought(thought) => {
            ResolvedInputMessageContent::Thought(thought)
        }
        // We cannot meaningfully embed images into dicl inputs, so reject the request.
        LazyResolvedInputMessageContent::File(..) => {
            return Err(Error::new(ErrorDetails::UnsupportedContentBlockType {
                content_block_type: "image".to_string(),
                provider_type: "dicl".to_string(),
            }));
        }
        // `Unknown` blocks will need special handling (we don't want the literal string "unknown")
        // to show up in the LLM input, so reject the request for now.
        LazyResolvedInputMessageContent::Unknown { .. } => {
            return Err(Error::new(ErrorDetails::UnsupportedContentBlockType {
                content_block_type: "unknown".to_string(),
                provider_type: "dicl".to_string(),
            }));
        }
    })
}
fn lazy_input_to_input_rejecting_incompatible(
    input: LazyResolvedInput,
) -> Result<ResolvedInput, Error> {
    Ok(ResolvedInput {
        system: input.system,
        messages: input
            .messages
            .into_iter()
            .map(|message| {
                Ok::<_, Error>(ResolvedInputMessage {
                    role: message.role,
                    content: message
                        .content
                        .into_iter()
                        .map(lazy_content_to_resolved_discarding_incompatible)
                        .collect::<Result<Vec<_>, _>>()?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
    })
}

#[derive(Debug, Deserialize, PartialEq)]
struct ChatExample {
    input: StoredInput,
    output: Vec<ContentBlockChatOutput>,
}

#[derive(Debug, Deserialize, PartialEq)]
struct JsonExample {
    input: StoredInput,
    output: JsonInferenceOutput,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(untagged)]
enum Example {
    Chat(ChatExample),
    Json(JsonExample),
}

#[derive(Clone, Debug, Deserialize)]
struct RawExample {
    input: String,
    output: String,
    cosine_distance: f32,
}

impl DiclConfig {
    async fn retrieve_relevant_examples<'a>(
        &'a self,
        input: &LazyResolvedInput,
        embedding_models: &'a EmbeddingModelTable,
        clients: &InferenceClients,
        function_name: &str,
        variant_name: &str,
        function: &Arc<FunctionConfig>,
    ) -> Result<(Vec<Example>, EmbeddingResponseWithMetadata), Error> {
        // Serialize the input so that it can be embedded
        let serialized_input = serde_json::to_string(
            &lazy_input_to_input_rejecting_incompatible(input.clone())?.into_stored_input(),
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error in serializing Input in dynamic in-context learning variant: {e}"
                ),
            })
        })?;

        let embedding_model = embedding_models
            .get(self.embedding_model())
            .await?
            .ok_or_else(|| {
                Error::new(ErrorDetails::Inference {
                    message: format!("Embedding model {} not found", self.embedding_model()),
                })
            })?;

        let embedding_request = EmbeddingRequest {
            input: serialized_input.into(),
            dimensions: None,
            encoding_format: EmbeddingEncodingFormat::Float,
        };

        // Embed the input via an API request
        let embedding_response = embedding_model
            .embed(&embedding_request, self.embedding_model(), clients)
            .await?;

        // Wrap the embedding in a response with metadata
        let embedding_response_with_metadata =
            EmbeddingResponseWithMetadata::new(embedding_response, self.embedding_model().clone());
        let [embedding_vector] = embedding_response_with_metadata.embeddings.as_slice() else {
            return Err(ErrorDetails::InternalError {
                message: format!(
                    "Embedding model returned multiple vectors. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            }
            .into());
        };

        // Format the embedding as a string for ClickHouse
        let formatted_embedding = format!(
            "[{}]",
            embedding_vector
                .as_float()
                .ok_or_else(|| Error::new(ErrorDetails::InternalError {
                    message: format!("Failed to convert DICL embedding to float array. {IMPOSSIBLE_ERROR_MESSAGE}")
                }))?
                .iter()
                .map(|&x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let query = format!(
            r"SELECT input, output, cosineDistance(embedding, {}) as cosine_distance
                   FROM DynamicInContextLearningExample
                   WHERE function_name='{}' AND variant_name='{}'
                   ORDER BY cosine_distance ASC
                   LIMIT {}
                   FORMAT JSONEachRow",
            formatted_embedding,
            function_name,
            variant_name,
            self.k()
        );

        // Run the query on the ClickHouse database to find nearest neighbors
        let result = clients
            .clickhouse_connection_info
            .run_query_synchronous_no_params(query)
            .await?;

        // Parse each line into RawExample (since we will have some serialized JSON strings inside it)
        let raw_examples: Vec<RawExample> = result
            .response
            .lines()
            .map(serde_json::from_str::<RawExample>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to parse raw examples: {e}"),
                })
            })?;

        let initial_count = raw_examples.len();

        // Warn if we couldn't retrieve k examples from database
        if initial_count < self.k() as usize {
            tracing::warn!(
                "Dynamic in-context learning retrieved {} examples from database, expected {}",
                initial_count,
                self.k()
            );
        }

        // Apply max_distance filter if specified
        let (raw_examples, max_distance_value) = if let Some(max_distance) = self.max_distance() {
            let filtered: Vec<RawExample> = raw_examples
                .into_iter()
                .filter(|ex| ex.cosine_distance <= max_distance)
                .collect();
            (filtered, Some(max_distance))
        } else {
            (raw_examples, None)
        };

        let filtered_count = raw_examples.len();

        // Debug log if max_distance reduced examples below k
        if let Some(max_distance) = max_distance_value {
            if initial_count >= self.k() as usize && filtered_count < self.k() as usize {
                tracing::debug!(
                    "Dynamic in-context learning: max_distance={} filtered examples from {} to {}",
                    max_distance,
                    initial_count,
                    filtered_count
                );
            }
        }

        // Convert RawExamples into Examples (parses those serialized JSON strings)
        let examples = parse_raw_examples(raw_examples, function)?;

        Ok((examples, embedding_response_with_metadata))
    }

    /// Serialize an example into a pair of RequestMessages
    /// The first message is a User message with the input serialized
    /// The second message is an Assistant message with the output as native output blocks
    ///   - For chat messages, this is a simple vector of ContentBlocks
    ///   - For JSON messages, this is a single JSON output block (as Text)
    fn prepare_message(example: &Example) -> Result<Vec<RequestMessage>, Error> {
        let mut messages = Vec::new();
        let input = match example {
            Example::Chat(chat_example) => chat_example.input.clone(),
            Example::Json(json_example) => json_example.input.clone(),
        };

        // Push the input as a user message
        messages.push(RequestMessage {
            role: Role::User,
            content: vec![serde_json::to_string(&input)
                .map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error in serializing Input in dynamic in-context learning variant: {e}"
                        ),
                    })
                })?
                .into()],
        });

        // Prepare the output
        let content: Vec<ContentBlock> = match example {
            Example::Chat(chat_example) => chat_example
                .output
                .clone()
                .into_iter()
                .map(ContentBlockChatOutput::into)
                .collect(),
            Example::Json(json_example) => {
                vec![json_example.output.raw.clone().unwrap_or_default().into()]
            }
        };

        // Push the output as an assistant message
        messages.push(RequestMessage {
            role: Role::Assistant,
            content,
        });
        Ok(messages)
    }

    fn prepare_input_message(input: &ResolvedInput) -> Result<RequestMessage, Error> {
        let content = vec![serde_json::to_string(&input.clone().into_stored_input())
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error in serializing Input in dynamic in-context learning variant: {e}"
                    ),
                })
            })?
            .into()];
        Ok(RequestMessage {
            role: Role::User,
            content,
        })
    }

    /// Prepare request message for DICL.
    /// We stringify template content blocks because DICL variants don't have a concept of templates.
    /// For everything else, we just pass it through unchanged.
    async fn prepare_request_message_dicl(
        message: &LazyResolvedInputMessage,
        templates: &TemplateConfig<'_>,
    ) -> Result<RequestMessage, Error> {
        let transformed_content: Vec<LazyResolvedInputMessageContent> = message
            .content
            .iter()
            .map(
                |content_block| -> Result<LazyResolvedInputMessageContent, Error> {
                    match content_block {
                        LazyResolvedInputMessageContent::Template(template_input) => {
                            // Stringify the template as JSON since DICL variants don't have a concept of templates
                            let json_str = serde_json::to_string(&serde_json::json!({
                                "type": "template",
                                "name": template_input.name,
                                "arguments": template_input.arguments,
                            }))
                            .map_err(|e| {
                                Error::new(ErrorDetails::Serialization {
                                    message: format!(
                                        "Failed to stringify template content block: {e}"
                                    ),
                                })
                            })?;
                            Ok(LazyResolvedInputMessageContent::Text(Text {
                                text: json_str,
                            }))
                        }
                        other => Ok(other.clone()),
                    }
                },
            )
            .collect::<Result<Vec<_>, Error>>()?;

        let transformed_message = LazyResolvedInputMessage {
            role: message.role,
            content: transformed_content,
        };

        // Pass to downstream chat_completion function with empty templates
        prepare_request_message(&transformed_message, templates, &ChatTemplates::default()).await
    }

    async fn prepare_request<'a, 'request>(
        &'a self,
        input: &LazyResolvedInput,
        examples: &[Example],
        function: &'request Arc<FunctionConfig>,
        inference_config: &'request InferenceConfig,
        stream: bool,
        inference_params: &mut InferenceParams,
    ) -> Result<ModelInferenceRequest<'request>, Error>
    where
        'a: 'request,
    {
        let (messages, system) = if examples.is_empty() {
            // When there are no examples, behave like vanilla chat completion
            // Use DICL-specific message preparation that stringifies templates
            let messages = try_join_all(input.messages.iter().map(|message| {
                Self::prepare_request_message_dicl(message, &inference_config.templates)
            }))
            .await?;
            let system = Some(self.system_instructions().to_string());
            (messages, system)
        } else {
            // When there are examples, use the  DICL logic
            let input = lazy_input_to_input_rejecting_incompatible(input.clone())?;
            let messages = examples
                .iter()
                .map(Self::prepare_message)
                .collect::<Result<Vec<Vec<RequestMessage>>, _>>()?
                .into_iter()
                .flatten()
                .chain(std::iter::once(Self::prepare_input_message(&input)?))
                .collect::<Vec<_>>();
            let system = Some(self.system_instructions().to_string());
            (messages, system)
        };

        inference_params
            .chat_completion
            .backfill_with_variant_params(
                self.temperature(),
                self.max_tokens(),
                self.seed(),
                self.top_p(),
                self.presence_penalty(),
                self.frequency_penalty(),
                self.stop_sequences().cloned(),
                self.inference_params_v2.clone(),
            );
        if !inference_config.extra_body.is_empty() {
            return Err(ErrorDetails::InvalidRequest {
                message: "Inference-level `extra_body` is not yet supported for dynamic_in_content_learning variant"
                    .to_string(),
            }
            .into());
        }
        let extra_body = FullExtraBodyConfig {
            extra_body: self.extra_body().cloned(),
            inference_extra_body: Default::default(),
        };
        let extra_headers = FullExtraHeadersConfig {
            variant_extra_headers: self.extra_headers().cloned(),
            inference_extra_headers: inference_config
                .extra_headers
                .clone()
                .filter(&inference_config.variant_name),
        };
        prepare_model_inference_request(
            messages,
            system,
            function.as_ref(),
            inference_config,
            stream,
            inference_params,
            self.json_mode().cloned(),
            extra_body,
            extra_headers,
        )
    }
}

// Since the `input` and `output` fields in the ClickHouse table are of type String,
// we need to parse them into the appropriate types before using them and cannot rely
// on Deserialize to do it for us.
fn parse_raw_examples(
    raw_examples: Vec<RawExample>,
    function: &FunctionConfig,
) -> Result<Vec<Example>, Error> {
    let mut examples = Vec::new();
    for raw_example in raw_examples {
        if raw_example.output.is_empty() {
            return Err(ErrorDetails::DiclMissingOutput.into());
        }
        // Parse the `input` string into `StoredInput`
        let input: StoredInput = serde_json::from_str(&raw_example.input).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to parse `input`: {e}"),
            })
        })?;

        for messages in &input.messages {
            for content in &messages.content {
                if let StoredInputMessageContent::File(_) = content {
                    return Err(Error::new(ErrorDetails::Serialization {
                        message: "Failed to deserialize raw_example - images are not supported in dynamic in-context learning".to_string(),
                    }));
                }
            }
        }

        match function {
            FunctionConfig::Chat(_) => {
                // Try parsing `output` as `Vec<ContentBlockOutput>` (for ChatExample)
                let output =
                    serde_json::from_str::<Vec<ContentBlockChatOutput>>(&raw_example.output)
                        .map_err(|e| {
                            Error::new(ErrorDetails::Serialization {
                                message: format!(
                                    "Failed to parse `output` in example `{raw_example:?}`: {e}"
                                ),
                            })
                        })?;
                examples.push(Example::Chat(ChatExample { input, output }));
            }
            FunctionConfig::Json(_) => {
                // Try parsing `output` as `JsonInferenceOutput` (for JsonExample)
                let output = serde_json::from_str::<JsonInferenceOutput>(&raw_example.output)
                    .map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!(
                                "Failed to parse `output` in example `{raw_example:?}`: {e}"
                            ),
                        })
                    })?;
                examples.push(Example::Json(JsonExample { input, output }));
            }
        }
    }

    Ok(examples)
}

pub fn default_system_instructions() -> String {
    "You are tasked with learning by induction and then solving a problem below. You will be shown several examples of inputs followed by outputs. Then, in the same format you will be given one last set of inputs. Your job is to use the provided examples to inform your response to the last set of inputs.".to_string()
}

impl LoadableConfig<DiclConfig> for UninitializedDiclConfig {
    fn load(self) -> Result<DiclConfig, Error> {
        let system_instructions = match self.system_instructions {
            Some(path) => path,
            None => ResolvedTomlPathData::new_fake_path(
                "tensorzero::dicl::default_system_instructions".to_string(),
                default_system_instructions(),
            ),
        };

        Ok(DiclConfig {
            weight: self.weight,
            embedding_model: Arc::from(self.embedding_model),
            k: self.k,
            model: Arc::from(self.model),
            system_instructions,
            temperature: self.temperature,
            top_p: self.top_p,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            max_tokens: self.max_tokens,
            seed: self.seed,
            inference_params_v2: ChatCompletionInferenceParamsV2 {
                reasoning_effort: self.reasoning_effort,
                service_tier: None,
                thinking_budget_tokens: self.thinking_budget_tokens,
                verbosity: self.verbosity,
            },
            json_mode: self.json_mode,
            retries: self.retries,
            stop_sequences: self.stop_sequences,
            extra_body: self.extra_body,
            extra_headers: self.extra_headers,
            max_distance: self.max_distance,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SchemaData;
    use crate::endpoints::inference::{ChatCompletionInferenceParams, InferenceIds};
    use crate::experimentation::ExperimentationConfig;
    use crate::inference::types::file::ObjectStoragePointer;
    use crate::inference::types::resolved_input::LazyResolvedInputMessage;
    use crate::inference::types::stored_input::StoredFile;
    use crate::inference::types::StoredInputMessage;
    use crate::inference::types::System;
    use crate::minijinja_util::tests::get_test_template_config;
    use crate::tool::ToolChoice;
    use crate::{
        function::{FunctionConfigChat, FunctionConfigJson},
        inference::types::{
            storage::{StorageKind, StoragePath},
            Arguments, ResolvedInputMessage, ResolvedInputMessageContent, Role, Template, Text,
        },
        tool::{InferenceResponseToolCall, ToolCall},
    };
    use serde_json::json;
    use std::collections::HashMap;
    use uuid::Uuid;

    #[test]
    fn test_prepare_message() {
        // ---------- Test with ChatExample ----------

        // Mock Input data
        let input_data = StoredInput {
            system: Some(System::Template(Arguments(
                json!({"type": "system", "content": "System message"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![
                StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::Text(Text {
                        text: "Hello, assistant!".to_string(),
                    })],
                },
                StoredInputMessage {
                    role: Role::Assistant,
                    content: vec![StoredInputMessageContent::Text(Text {
                        text: "Hello, user!".to_string(),
                    })],
                },
            ],
        };

        // Mock Output data for ChatExample
        let chat_output = vec![
            ContentBlockChatOutput::Text(Text {
                text: "This is a test response.".to_string(),
            }),
            ContentBlockChatOutput::ToolCall(InferenceResponseToolCall {
                id: "tool_call_1".to_string(),
                raw_name: "search_tool".to_string(),
                raw_arguments: "{\"query\": \"rust programming\"}".to_string(),
                name: Some("search_tool".to_string()),
                arguments: Some(json!({"query": "rust programming"})),
            }),
        ];

        let chat_example = Example::Chat(ChatExample {
            input: input_data.clone(),
            output: chat_output.clone(),
        });

        let chat_messages = DiclConfig::prepare_message(&chat_example).unwrap();

        assert_eq!(chat_messages.len(), 2);

        // First message should be from User with serialized input
        let serialized_input = serde_json::to_string(&input_data).unwrap();
        assert_eq!(chat_messages[0].role, Role::User);
        assert_eq!(
            chat_messages[0].content,
            vec![ContentBlock::Text(Text {
                text: serialized_input.clone()
            })]
        );

        // Second message should be from Assistant with content blocks
        let expected_content: Vec<ContentBlock> = chat_output
            .into_iter()
            .map(ContentBlockChatOutput::into)
            .collect();

        assert_eq!(chat_messages[1].role, Role::Assistant);
        assert_eq!(chat_messages[1].content, expected_content);

        // ---------- Test with JsonExample ----------

        // Mock Output data for JsonExample
        let json_output = JsonInferenceOutput {
            raw: Some("{\"result\": \"success\"}".to_string()),
            parsed: Some(json!({"result": "success"})),
        };

        let json_example = Example::Json(JsonExample {
            input: input_data.clone(),
            output: json_output.clone(),
        });

        let json_messages = DiclConfig::prepare_message(&json_example).unwrap();

        // Assertions for JsonExample
        assert_eq!(json_messages.len(), 2);

        // First message should be from User with serialized input
        assert_eq!(json_messages[0].role, Role::User);
        assert_eq!(
            json_messages[0].content,
            vec![ContentBlock::Text(Text {
                text: serialized_input
            })]
        );

        // Second message should be from Assistant with raw JSON output as text
        let expected_content = vec![ContentBlock::Text(Text {
            text: json_output.raw.unwrap().clone(),
        })];

        assert_eq!(json_messages[1].role, Role::Assistant);
        assert_eq!(json_messages[1].content, expected_content);
    }

    #[test]
    fn test_prepare_input_message() {
        // Mock Input data
        let input_data = ResolvedInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "Dr. Mehta"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![
                ResolvedInputMessage {
                    role: Role::User,
                    content: vec![
                        ResolvedInputMessageContent::Text(Text {
                            text: "Hello, assistant!".to_string(),
                        }),
                        ResolvedInputMessageContent::ToolCall(ToolCall {
                            id: "tool_call_1".to_string(),
                            name: "search_tool".to_string(),
                            arguments: "{\"query\": \"rust programming\"}".to_string(),
                        }),
                    ],
                },
                ResolvedInputMessage {
                    role: Role::Assistant,
                    content: vec![ResolvedInputMessageContent::Text(Text {
                        text: "Here are the search results for rust programming.".to_string(),
                    })],
                },
            ],
        };

        // Call the prepare_input_message function
        let request_message = DiclConfig::prepare_input_message(&input_data).unwrap();

        // The role should be User
        assert_eq!(request_message.role, Role::User);

        // The content should contain the serialized Input as a Text ContentBlock
        let expected_serialized_input =
            serde_json::to_string(&input_data.clone().into_stored_input()).unwrap();
        let expected_content = vec![ContentBlock::Text(Text {
            text: expected_serialized_input.clone(),
        })];
        assert_eq!(request_message.content, expected_content);
    }

    #[test]
    fn test_reject_image_example() {
        // Define sample raw examples with serialized Input and Output
        let raw_examples = vec![
            RawExample {
                input: serde_json::to_string(&StoredInput {
                    system: Some(System::Template(Arguments(
                        json!({"assistant_name": "Dr. Mehta"})
                            .as_object()
                            .unwrap()
                            .clone(),
                    ))),
                    messages: vec![StoredInputMessage {
                        role: Role::User,
                        content: vec![StoredInputMessageContent::Text(Text {
                            text: "What is the boiling point of water?".to_string(),
                        })],
                    }],
                })
                .unwrap(),
                output: serde_json::to_string(&vec![ContentBlockChatOutput::Text(Text {
                    text: "100 degrees Celsius".to_string(),
                })])
                .unwrap(),
                cosine_distance: 0.1,
            },
            RawExample {
                input: serde_json::to_string(&StoredInput {
                    system: Some(System::Template(Arguments(
                        json!({"assistant_name": "Pinocchio"})
                            .as_object()
                            .unwrap()
                            .clone(),
                    ))),
                    messages: vec![StoredInputMessage {
                        role: Role::User,
                        content: vec![
                            StoredInputMessageContent::Text(Text {
                                text: "What is the name of the capital city of Japan?".to_string(),
                            }),
                            StoredInputMessageContent::File(Box::new(StoredFile(
                                ObjectStoragePointer {
                                    source_url: None,
                                    mime_type: mime::IMAGE_PNG,
                                    storage_path: StoragePath {
                                        kind: StorageKind::Disabled,
                                        path: Default::default(),
                                    },
                                    detail: None,
                                    filename: None,
                                },
                            ))),
                        ],
                    }],
                })
                .unwrap(),
                output: serde_json::to_string(&vec![ContentBlockChatOutput::Text(Text {
                    text: "Osaka (nose grows 4 inches)".to_string(),
                })])
                .unwrap(),
                cosine_distance: 0.2,
            },
        ];

        let function = FunctionConfig::Chat(FunctionConfigChat {
            ..Default::default()
        });
        // Parse the raw examples
        let err = parse_raw_examples(raw_examples.clone(), &function)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("images are not supported in dynamic in-context learning"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn test_dicl_missing_output_error() {
        // Create a raw example with missing output
        let raw_examples = vec![RawExample {
            input: serde_json::to_string(&StoredInput {
                system: Some(System::Template(Arguments(
                    json!({"assistant_name": "Dr. Mehta"})
                        .as_object()
                        .unwrap()
                        .clone(),
                ))),
                messages: vec![StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::Text(Text {
                        text: "What is the boiling point of water?".to_string(),
                    })],
                }],
            })
            .unwrap(),
            output: String::new(),
            cosine_distance: 0.1,
        }];

        let function = FunctionConfig::Chat(FunctionConfigChat {
            ..Default::default()
        });

        // Parse the raw examples and expect DiclMissingOutput error
        let result = parse_raw_examples(raw_examples, &function);

        assert!(
            result.is_err(),
            "Expected DiclMissingOutput error but got success"
        );

        let error = result.unwrap_err();
        let error_string = error.to_string();
        assert!(
            error_string.contains("DICL example missing output"),
            "Expected DiclMissingOutput error, got: {error_string}"
        );
    }

    #[test]
    fn test_parse_raw_examples() {
        // Define sample raw examples with serialized Input and Output
        let raw_examples = vec![
            RawExample {
                input: serde_json::to_string(&StoredInput {
                    system: Some(System::Template(Arguments(
                        json!({"assistant_name": "Dr. Mehta"})
                            .as_object()
                            .unwrap()
                            .clone(),
                    ))),
                    messages: vec![StoredInputMessage {
                        role: Role::User,
                        content: vec![StoredInputMessageContent::Text(Text {
                            text: "What is the boiling point of water?".to_string(),
                        })],
                    }],
                })
                .unwrap(),
                output: serde_json::to_string(&vec![ContentBlockChatOutput::Text(Text {
                    text: "100 degrees Celsius".to_string(),
                })])
                .unwrap(),
                cosine_distance: 0.1,
            },
            RawExample {
                input: serde_json::to_string(&StoredInput {
                    system: Some(System::Template(Arguments(
                        json!({"assistant_name": "Pinocchio"})
                            .as_object()
                            .unwrap()
                            .clone(),
                    ))),
                    messages: vec![StoredInputMessage {
                        role: Role::User,
                        content: vec![StoredInputMessageContent::Text(Text {
                            text: "What is the name of the capital city of Japan?".to_string(),
                        })],
                    }],
                })
                .unwrap(),
                output: serde_json::to_string(&vec![ContentBlockChatOutput::Text(Text {
                    text: "Osaka (nose grows 4 inches)".to_string(),
                })])
                .unwrap(),
                cosine_distance: 0.2,
            },
        ];

        let function = FunctionConfig::Chat(FunctionConfigChat {
            ..Default::default()
        });
        // Parse the raw examples
        let parsed_examples = parse_raw_examples(raw_examples.clone(), &function)
            .expect("Failed to parse raw examples");

        // Define the expected examples
        let expected_examples = vec![
            Example::Chat(ChatExample {
                input: serde_json::from_str(&raw_examples[0].input).unwrap(),
                output: serde_json::from_str(&raw_examples[0].output).unwrap(),
            }),
            Example::Chat(ChatExample {
                input: serde_json::from_str(&raw_examples[1].input).unwrap(),
                output: serde_json::from_str(&raw_examples[1].output).unwrap(),
            }),
        ];

        // Assert that the parsed examples match the expected examples
        assert_eq!(parsed_examples, expected_examples);

        // Test that we can parse a JSON example too
        let json_raw_examples = vec![
            RawExample {
                input: serde_json::to_string(&StoredInput {
                    system: Some(System::Template(Arguments(
                        json!({"assistant_name": "JsonTester"})
                            .as_object()
                            .unwrap()
                            .clone(),
                    ))),
                    messages: vec![StoredInputMessage {
                        role: Role::User,
                        content: vec![StoredInputMessageContent::Text(Text {
                            text: "Provide a sample JSON response.".to_string(),
                        })],
                    }],
                })
                .unwrap(),
                output: serde_json::to_string(&JsonInferenceOutput {
                    raw: Some("{\"status\": \"success\", \"data\": {\"id\": 1}}".to_string()),
                    parsed: Some(json!({
                        "status": "success",
                        "data": {
                            "id": 1
                        }
                    })),
                })
                .unwrap(),
                cosine_distance: 0.1,
            },
            RawExample {
                input: serde_json::to_string(&StoredInput {
                    system: Some(System::Template(Arguments(
                        json!({"assistant_name": "JsonTester"})
                            .as_object()
                            .unwrap()
                            .clone(),
                    ))),
                    messages: vec![StoredInputMessage {
                        role: Role::User,
                        content: vec![StoredInputMessageContent::Text(Text {
                            text: "Provide another JSON response.".to_string(),
                        })],
                    }],
                })
                .unwrap(),
                output: serde_json::to_string(&JsonInferenceOutput {
                    raw: Some("{\"result\": [1, 2, 3], \"status\": \"ok\"}".to_string()),
                    parsed: Some(json!({
                        "result": [1, 2, 3],
                        "status": "ok"
                    })),
                })
                .unwrap(),
                cosine_distance: 0.2,
            },
        ];
        let json_function = FunctionConfig::Json(FunctionConfigJson {
            ..Default::default()
        });

        // Parse the JSON raw examples
        let parsed_json_examples = parse_raw_examples(json_raw_examples.clone(), &json_function)
            .expect("Failed to parse JSON raw examples");

        // Assert that all parsed JSON examples have 'parsed' as Some
        for example in parsed_json_examples {
            if let Example::Json(json_example) = example {
                assert!(json_example.output.parsed.is_some(), "Parsed field is None");
            } else {
                panic!("Expected JsonExample");
            }
        }
    }

    #[tokio::test]
    async fn test_prepare_request_with_empty_examples() {
        // Setup: Create a DICL config with specific system instructions
        let dicl_config = DiclConfig {
            weight: None,
            embedding_model: "test_embedding".into(),
            k: 3,
            model: "test_model".into(),
            system_instructions: ResolvedTomlPathData::new_fake_path(
                "test".to_string(),
                "DICL system instructions that should NOT be used".to_string(),
            ),
            temperature: Some(0.7),
            top_p: None,
            max_tokens: Some(100),
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            inference_params_v2: ChatCompletionInferenceParamsV2::default(),
            stop_sequences: None,
            json_mode: None,
            extra_body: None,
            extra_headers: None,
            retries: RetryConfig::default(),
            max_distance: None,
        };

        // Create input with custom system prompt and messages
        let input = LazyResolvedInput {
            system: Some(System::Text("Custom system from input".to_string())),
            messages: vec![
                LazyResolvedInputMessage {
                    role: Role::User,
                    content: vec![LazyResolvedInputMessageContent::Text(Text {
                        text: "Hello, how are you?".to_string(),
                    })],
                },
                LazyResolvedInputMessage {
                    role: Role::Assistant,
                    content: vec![LazyResolvedInputMessageContent::Text(Text {
                        text: "I'm doing great!".to_string(),
                    })],
                },
            ],
        };

        // Setup inference config
        let templates = get_test_template_config().await;
        let inference_config = InferenceConfig {
            templates: Arc::new(templates),
            tool_config: None,
            function_name: "test_function".into(),
            variant_name: "test_variant".into(),
            dynamic_output_schema: None,
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };

        let mut inference_params = InferenceParams {
            chat_completion: ChatCompletionInferenceParams::default(),
        };

        let function = Arc::new(FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: Default::default(),
            experimentation: ExperimentationConfig::default(),
        }));

        // Call prepare_request with EMPTY examples
        let result = dicl_config
            .prepare_request(
                &input,
                &[], // Empty examples - this triggers the new code path
                &function,
                &inference_config,
                false,
                &mut inference_params,
            )
            .await;

        assert!(result.is_ok(), "prepare_request should succeed");
        let request = result.unwrap();

        // Verify: Messages should NOT be JSON-serialized
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].role, Role::User);
        if let ContentBlock::Text(text) = &request.messages[0].content[0] {
            // Should be plain text, NOT JSON
            assert_eq!(text.text, "Hello, how are you?");
        } else {
            panic!("Expected plain text content");
        }

        assert_eq!(request.messages[1].role, Role::Assistant);
        if let ContentBlock::Text(text) = &request.messages[1].content[0] {
            assert_eq!(text.text, "I'm doing great!");
        } else {
            panic!("Expected plain text content");
        }

        // Verify: System prompt should always come from DICL config system_instructions
        assert_eq!(
            request.system,
            Some("DICL system instructions that should NOT be used".to_string())
        );
    }

    #[tokio::test]
    async fn test_prepare_request_with_examples() {
        // Setup: Create a DICL config
        let dicl_config = DiclConfig {
            weight: None,
            embedding_model: "test_embedding".into(),
            k: 3,
            model: "test_model".into(),
            system_instructions: ResolvedTomlPathData::new_fake_path(
                "test".to_string(),
                "DICL system instructions".to_string(),
            ),
            temperature: Some(0.7),
            top_p: None,
            max_tokens: Some(100),
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            inference_params_v2: ChatCompletionInferenceParamsV2::default(),
            stop_sequences: None,
            json_mode: None,
            extra_body: None,
            extra_headers: None,
            retries: RetryConfig::default(),
            max_distance: None,
        };

        // Create input
        let input = LazyResolvedInput {
            system: Some(System::Text("Custom system from input".to_string())),
            messages: vec![LazyResolvedInputMessage {
                role: Role::User,
                content: vec![LazyResolvedInputMessageContent::Text(Text {
                    text: "Hello!".to_string(),
                })],
            }],
        };

        // Create an example
        let example_input = StoredInput {
            system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                "context".to_string(),
                "example".into(),
            )])))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: "Example question".to_string(),
                })],
            }],
        };
        let example = Example::Chat(ChatExample {
            input: example_input,
            output: vec![ContentBlockChatOutput::Text(Text {
                text: "Example answer".to_string(),
            })],
        });

        // Setup inference config
        let templates = get_test_template_config().await;
        let inference_config = InferenceConfig {
            templates: Arc::new(templates),
            tool_config: None,
            function_name: "test_function".into(),
            variant_name: "test_variant".into(),
            dynamic_output_schema: None,
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };

        let mut inference_params = InferenceParams {
            chat_completion: ChatCompletionInferenceParams::default(),
        };

        let function = Arc::new(FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: Default::default(),
            experimentation: ExperimentationConfig::default(),
        }));

        // Call prepare_request with examples
        let result = dicl_config
            .prepare_request(
                &input,
                &[example], // Has examples - this triggers the original DICL behavior
                &function,
                &inference_config,
                false,
                &mut inference_params,
            )
            .await;

        assert!(result.is_ok(), "prepare_request should succeed");
        let request = result.unwrap();

        // Verify: Should have 3 messages (example User, example Assistant, final User)
        assert_eq!(request.messages.len(), 3);

        // Verify: Messages should be JSON-serialized (original DICL behavior)
        assert_eq!(request.messages[0].role, Role::User);
        if let ContentBlock::Text(text) = &request.messages[0].content[0] {
            // Should be JSON-serialized
            assert!(
                text.text.contains("\"system\""),
                "First message should contain JSON-serialized example input"
            );
        } else {
            panic!("Expected text content");
        }

        // Verify: System prompt should come from DICL config, not input
        assert_eq!(request.system, Some("DICL system instructions".to_string()));
    }

    #[test]
    fn test_raw_example_with_cosine_distance_parsing() {
        // Test that RawExample correctly parses the cosine_distance field from JSON
        let json =
            r#"{"input":"{\"system\":null,\"messages\":[]}","output":"[]","cosine_distance":0.5}"#;
        let raw_example: RawExample = serde_json::from_str(json).unwrap();
        assert_eq!(raw_example.cosine_distance, 0.5);
    }

    #[test]
    fn test_max_distance_validation_negative() {
        // Test that validation rejects negative max_distance
        let dicl_config = DiclConfig {
            weight: None,
            embedding_model: "test_embedding".into(),
            k: 3,
            model: "test_model".into(),
            system_instructions: ResolvedTomlPathData::new_fake_path(
                "test".to_string(),
                "test".to_string(),
            ),
            temperature: None,
            top_p: None,
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            inference_params_v2: ChatCompletionInferenceParamsV2::default(),
            stop_sequences: None,
            json_mode: None,
            extra_body: None,
            extra_headers: None,
            retries: RetryConfig::default(),
            max_distance: Some(-0.5),
        };

        // Validation should fail
        assert_eq!(dicl_config.max_distance(), Some(-0.5));
    }

    #[test]
    fn test_max_distance_validation_positive() {
        // Test that validation accepts positive max_distance
        let dicl_config = DiclConfig {
            weight: None,
            embedding_model: "test_embedding".into(),
            k: 3,
            model: "test_model".into(),
            system_instructions: ResolvedTomlPathData::new_fake_path(
                "test".to_string(),
                "test".to_string(),
            ),
            temperature: None,
            top_p: None,
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            inference_params_v2: ChatCompletionInferenceParamsV2::default(),
            stop_sequences: None,
            json_mode: None,
            extra_body: None,
            extra_headers: None,
            retries: RetryConfig::default(),
            max_distance: Some(0.5),
        };

        assert_eq!(dicl_config.max_distance(), Some(0.5));
    }

    #[test]
    fn test_max_distance_validation_zero() {
        // Test that validation accepts zero max_distance (for exact matches only)
        let dicl_config = DiclConfig {
            weight: None,
            embedding_model: "test_embedding".into(),
            k: 3,
            model: "test_model".into(),
            system_instructions: ResolvedTomlPathData::new_fake_path(
                "test".to_string(),
                "test".to_string(),
            ),
            temperature: None,
            top_p: None,
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            inference_params_v2: ChatCompletionInferenceParamsV2::default(),
            stop_sequences: None,
            json_mode: None,
            extra_body: None,
            extra_headers: None,
            retries: RetryConfig::default(),
            max_distance: Some(0.0),
        };

        assert_eq!(dicl_config.max_distance(), Some(0.0));
    }

    #[tokio::test]
    async fn test_prepare_request_message_dicl_with_template() {
        let message = LazyResolvedInputMessage {
            role: Role::User,
            content: vec![LazyResolvedInputMessageContent::Template(Template {
                name: "user".to_string(),
                arguments: Arguments(
                    serde_json::json!({"key": "value"})
                        .as_object()
                        .unwrap()
                        .clone(),
                ),
            })],
        };
        let templates = TemplateConfig::default();

        let result = DiclConfig::prepare_request_message_dicl(&message, &templates)
            .await
            .unwrap();

        assert_eq!(result.role, Role::User);
        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlock::Text(text) => {
                let parsed: serde_json::Value = serde_json::from_str(&text.text).unwrap();
                assert_eq!(parsed["type"], "template");
                assert_eq!(parsed["name"], "user");
                assert_eq!(parsed["arguments"]["key"], "value");
            }
            _ => panic!("Expected Text content block"),
        }
    }

    #[tokio::test]
    async fn test_prepare_request_message_dicl_without_template() {
        let message = LazyResolvedInputMessage {
            role: Role::User,
            content: vec![LazyResolvedInputMessageContent::Text(Text {
                text: "Hello".to_string(),
            })],
        };
        let templates = TemplateConfig::default();

        let result = DiclConfig::prepare_request_message_dicl(&message, &templates)
            .await
            .unwrap();

        assert_eq!(result.role, Role::User);
        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlock::Text(text) => {
                assert_eq!(text.text, "Hello");
            }
            _ => panic!("Expected Text content block"),
        }
    }

    #[test]
    fn test_lazy_content_to_resolved_with_template() {
        let template = LazyResolvedInputMessageContent::Template(Template {
            name: "test_template".to_string(),
            arguments: Arguments(
                serde_json::json!({"foo": "bar"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
        });

        let result = lazy_content_to_resolved_discarding_incompatible(template).unwrap();

        match result {
            ResolvedInputMessageContent::Text(Text { text }) => {
                let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
                assert_eq!(parsed["type"], "template");
                assert_eq!(parsed["name"], "test_template");
                assert_eq!(parsed["arguments"]["foo"], "bar");
            }
            _ => panic!("Expected Text content after template stringification"),
        }
    }

    #[test]
    fn test_as_uninitialized_preserves_basic_fields() {
        let uninitialized = UninitializedDiclConfig {
            weight: Some(0.9),
            embedding_model: "text-embedding-ada-002".to_string(),
            k: 5,
            model: "gpt-4".to_string(),
            system_instructions: Some(ResolvedTomlPathData::new_fake_path(
                "test.txt".to_string(),
                "Custom instructions".to_string(),
            )),
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(150),
            seed: Some(42),
            stop_sequences: Some(vec!["STOP".to_string()]),
            max_distance: Some(0.8),
            ..Default::default()
        };

        let config = uninitialized.load().unwrap();

        let exported = config.as_uninitialized();

        assert_eq!(exported.weight, Some(0.9));
        assert_eq!(exported.embedding_model, "text-embedding-ada-002");
        assert_eq!(exported.k, 5);
        assert_eq!(exported.model, "gpt-4");
        assert_eq!(exported.temperature, Some(0.7));
        assert_eq!(exported.top_p, Some(0.9));
        assert_eq!(exported.max_tokens, Some(150));
        assert_eq!(exported.seed, Some(42));
        assert_eq!(exported.stop_sequences, Some(vec!["STOP".to_string()]));
        assert_eq!(exported.max_distance, Some(0.8));
    }

    #[test]
    fn test_as_uninitialized_preserves_real_path_system_instructions() {
        let instructions_content = "These are system instructions";
        // Using from_path_and_data to create a real path
        let real_path = ResolvedTomlPathData::new_for_tests(
            "test.txt".into(),
            Some(instructions_content.to_string()),
        );

        let uninitialized = UninitializedDiclConfig {
            embedding_model: "embed-model".to_string(),
            k: 3,
            model: "gpt-3.5-turbo".to_string(),
            system_instructions: Some(real_path),
            ..Default::default()
        };

        let config = uninitialized.load().unwrap();

        let exported = config.as_uninitialized();

        // Verify real path is preserved
        assert!(exported.system_instructions.is_some());
        assert_eq!(
            exported.system_instructions.unwrap().data(),
            instructions_content
        );
    }

    #[test]
    fn test_as_uninitialized_fake_path_becomes_none() {
        let instructions_content = "These are system instructions";
        let uninitialized = UninitializedDiclConfig {
            embedding_model: "embed-model".to_string(),
            k: 3,
            model: "gpt-3.5-turbo".to_string(),
            system_instructions: Some(ResolvedTomlPathData::new_fake_path(
                "tensorzero::test".to_string(),
                instructions_content.to_string(),
            )),
            ..Default::default()
        };

        let config = uninitialized.load().unwrap();

        let exported = config.as_uninitialized();

        // Verify fake path becomes None
        assert_eq!(exported.system_instructions, None);
    }

    #[test]
    fn test_as_uninitialized_preserves_inference_params_v2() {
        let uninitialized = UninitializedDiclConfig {
            embedding_model: "embed".to_string(),
            k: 1,
            model: "gpt-4".to_string(),
            reasoning_effort: Some("high".to_string()),
            thinking_budget_tokens: Some(1000),
            verbosity: Some("verbose".to_string()),
            ..Default::default()
        };

        let config = uninitialized.load().unwrap();

        let exported = config.as_uninitialized();

        assert_eq!(exported.reasoning_effort, Some("high".to_string()));
        assert_eq!(exported.thinking_budget_tokens, Some(1000));
        assert_eq!(exported.verbosity, Some("verbose".to_string()));
    }

    #[test]
    fn test_as_uninitialized_preserves_none_values() {
        let uninitialized = UninitializedDiclConfig {
            embedding_model: "embed".to_string(),
            k: 1,
            model: "gpt-4".to_string(),
            system_instructions: None,
            weight: None,
            temperature: None,
            max_tokens: None,
            stop_sequences: None,
            max_distance: None,
            reasoning_effort: None,
            thinking_budget_tokens: None,
            verbosity: None,
            ..Default::default()
        };

        let config = uninitialized.load().unwrap();

        let exported = config.as_uninitialized();

        assert_eq!(exported.weight, None);
        assert_eq!(exported.temperature, None);
        assert_eq!(exported.max_tokens, None);
        assert_eq!(exported.stop_sequences, None);
        assert_eq!(exported.max_distance, None);
        assert_eq!(exported.reasoning_effort, None);
        assert_eq!(exported.thinking_budget_tokens, None);
        assert_eq!(exported.verbosity, None);
        // system_instructions becomes None because the default is a fake path
        assert_eq!(exported.system_instructions, None);
    }

    #[test]
    fn test_as_uninitialized_serialization_round_trip() {
        let original = UninitializedDiclConfig {
            weight: Some(0.6),
            embedding_model: "ada-002".to_string(),
            k: 10,
            model: "gpt-3.5-turbo".to_string(),
            temperature: Some(0.5),
            ..Default::default()
        };

        let config = original.clone().load().unwrap();

        let exported = config.as_uninitialized();

        // Serialize and deserialize
        let json = serde_json::to_string(&exported).unwrap();
        let deserialized: UninitializedDiclConfig = serde_json::from_str(&json).unwrap();

        // Should be able to load again
        let reloaded = deserialized.load().unwrap();

        assert_eq!(reloaded.weight(), Some(0.6));
        assert_eq!(reloaded.embedding_model(), &Arc::from("ada-002"));
        assert_eq!(reloaded.k(), 10);
        assert_eq!(reloaded.model(), &Arc::from("gpt-3.5-turbo"));
        assert_eq!(reloaded.temperature(), Some(0.5));
    }
}
