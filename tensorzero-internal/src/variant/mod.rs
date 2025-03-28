use backon::ExponentialBuilder;
use backon::Retryable;
use futures::StreamExt;
use itertools::izip;
use serde::Deserialize;
use serde::Serialize;
use std::borrow::Cow;
use std::sync::Arc;
use tokio::time::Duration;
use tracing::instrument;
use uuid::Uuid;

use crate::config_parser::PathWithContents;
use crate::embeddings::EmbeddingModelTable;
use crate::endpoints::inference::InferenceIds;
use crate::endpoints::inference::{InferenceClients, InferenceModels, InferenceParams};
use crate::error::Error;
use crate::error::ErrorDetails;
use crate::function::FunctionConfig;
use crate::inference::types::batch::StartBatchModelInferenceWithMetadata;
use crate::inference::types::extra_body::FullExtraBodyConfig;
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::ResolvedInput;
use crate::inference::types::{
    FunctionType, InferenceResultChunk, InferenceResultStream, ModelInferenceRequest,
    ModelInferenceResponseWithMetadata, RequestMessage,
};
use crate::jsonschema_util::DynamicJSONSchema;
use crate::minijinja_util::TemplateConfig;
use crate::model::ModelTable;
use crate::model::StreamResponse;
use crate::tool::{create_dynamic_implicit_tool_config, ToolCallConfig};
use crate::{inference::types::InferenceResult, model::ModelConfig};

pub mod best_of_n_sampling;
pub mod chat_completion;
pub mod dicl;
pub mod mixture_of_n;

#[derive(Debug)]
pub enum VariantConfig {
    ChatCompletion(chat_completion::ChatCompletionConfig),
    BestOfNSampling(best_of_n_sampling::BestOfNSamplingConfig),
    Dicl(dicl::DiclConfig),
    MixtureOfN(mixture_of_n::MixtureOfNConfig),
}

/// This type is used to determine how to enforce JSON mode for a given variant.
/// Variants represent JSON mode in a slightly more abstract sense than ModelInferenceRequests, as
/// we support coercing tool calls into JSON mode.
/// This is represented as a tool config in the
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JsonMode {
    Off,
    On,
    Strict,
    ImplicitTool,
}

/// Configuration that applies to the current inference request.
#[derive(Clone, Debug)]
pub struct InferenceConfig<'a, 'request> {
    pub tool_config: Option<&'request ToolCallConfig>,
    pub templates: &'request TemplateConfig<'a>,
    pub dynamic_output_schema: Option<&'request DynamicJSONSchema>,
    pub function_name: &'request str,
    pub variant_name: Option<&'request str>,
    pub ids: InferenceIds,
    pub extra_body: UnfilteredInferenceExtraBody,
    /// Optional arbitrary data, only used when constructing the cache key.
    /// This is used by best_of_n/mixture_of_n to force different sub-variants
    /// to have different cache keys.
    /// This field should only ever be forwarded to `ModelInferenceRequest`
    pub extra_cache_key: Option<String>,
}

/// Maps to the subset of Config that applies to the current inference request.
#[derive(Clone, Debug)]
pub struct BatchInferenceConfig<'a> {
    pub tool_configs: Vec<Option<ToolCallConfig>>,
    pub templates: &'a TemplateConfig<'a>,
    pub dynamic_output_schemas: Vec<Option<DynamicJSONSchema>>,
    pub function_name: &'a str,
    pub variant_name: Option<&'a str>,
}
impl<'a> BatchInferenceConfig<'a> {
    pub fn inference_configs(
        &'a self,
        episode_ids: &[Uuid],
        inference_ids: &[Uuid],
    ) -> Vec<InferenceConfig<'a, 'a>> {
        izip!(
            self.tool_configs.iter().map(|x| x.as_ref()),
            self.dynamic_output_schemas.iter().map(|x| x.as_ref()),
            episode_ids.iter(),
            inference_ids.iter()
        )
        .map(
            |(tool_config, dynamic_output_schema, episode_id, inference_id)| InferenceConfig {
                templates: self.templates,
                tool_config,
                dynamic_output_schema,
                function_name: self.function_name,
                variant_name: self.variant_name,
                ids: InferenceIds {
                    inference_id: *inference_id,
                    episode_id: *episode_id,
                },
                // Not yet supported for batch inference requests
                extra_body: Default::default(),
                extra_cache_key: None,
            },
        )
        .collect()
    }
}

#[derive(Debug)]
pub struct ModelUsedInfo {
    pub model_name: Arc<str>,
    pub model_provider_name: Arc<str>,
    pub raw_request: String,
    pub system: Option<String>,
    pub input_messages: Vec<RequestMessage>,
    pub inference_params: InferenceParams,
    pub previous_model_inference_results: Vec<ModelInferenceResponseWithMetadata>,
    pub cached: bool,
}

pub trait Variant {
    async fn infer<'a: 'request, 'request>(
        &self,
        input: &ResolvedInput,
        models: &'request InferenceModels<'a>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'static, 'request>,
        clients: &'request InferenceClients<'request>,
        inference_params: InferenceParams,
    ) -> Result<InferenceResult, Error>;

    async fn infer_stream<'request>(
        &self,
        input: &ResolvedInput,
        models: &'request InferenceModels<'_>,
        function: &FunctionConfig,
        inference_config: &'request InferenceConfig<'static, 'request>,
        clients: &'request InferenceClients<'request>,
        inference_params: InferenceParams,
    ) -> Result<(InferenceResultStream, ModelUsedInfo), Error>;

    fn validate(
        &self,
        function: &FunctionConfig,
        models: &mut ModelTable,
        embedding_models: &EmbeddingModelTable,
        templates: &TemplateConfig,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(), Error>;

    fn get_all_template_paths(&self) -> Vec<&PathWithContents>;

    async fn start_batch_inference<'a>(
        &'a self,
        input: &[ResolvedInput],
        models: &'a InferenceModels<'a>,
        function: &'a FunctionConfig,
        inference_configs: &'a [InferenceConfig<'a, 'a>],
        clients: &'a InferenceClients<'a>,
        inference_params: Vec<InferenceParams>,
    ) -> Result<StartBatchModelInferenceWithMetadata<'a>, Error>;
}

impl VariantConfig {
    pub fn weight(&self) -> Option<f64> {
        match self {
            VariantConfig::ChatCompletion(params) => params.weight,
            VariantConfig::BestOfNSampling(params) => params.weight,
            VariantConfig::Dicl(params) => params.weight,
            VariantConfig::MixtureOfN(params) => params.weight,
        }
    }
}

impl Variant for VariantConfig {
    #[instrument(
        fields(function_name = %inference_config.function_name, variant_name = %inference_config.variant_name.unwrap_or("")),
        skip_all
    )]
    async fn infer<'a: 'request, 'request>(
        &self,
        input: &ResolvedInput,
        models: &'request InferenceModels<'a>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'static, 'request>,
        clients: &'request InferenceClients<'request>,
        inference_params: InferenceParams,
    ) -> Result<InferenceResult, Error> {
        match self {
            VariantConfig::ChatCompletion(params) => {
                params
                    .infer(
                        input,
                        models,
                        function,
                        inference_config,
                        clients,
                        inference_params,
                    )
                    .await
            }
            VariantConfig::BestOfNSampling(params) => {
                params
                    .infer(
                        input,
                        models,
                        function,
                        inference_config,
                        clients,
                        inference_params,
                    )
                    .await
            }

            VariantConfig::Dicl(params) => {
                params
                    .infer(
                        input,
                        models,
                        function,
                        inference_config,
                        clients,
                        inference_params,
                    )
                    .await
            }
            VariantConfig::MixtureOfN(params) => {
                params
                    .infer(
                        input,
                        models,
                        function,
                        inference_config,
                        clients,
                        inference_params,
                    )
                    .await
            }
        }
    }

    #[instrument(
        fields(function_name = %inference_config.function_name, variant_name = %inference_config.variant_name.unwrap_or("")),
        skip_all
    )]
    async fn infer_stream<'a, 'request>(
        &self,
        input: &ResolvedInput,
        models: &'request InferenceModels<'a>,
        function: &FunctionConfig,
        inference_config: &'request InferenceConfig<'static, 'request>,
        clients: &'request InferenceClients<'request>,
        inference_params: InferenceParams,
    ) -> Result<(InferenceResultStream, ModelUsedInfo), Error> {
        match self {
            VariantConfig::ChatCompletion(params) => {
                params
                    .infer_stream(
                        input,
                        models,
                        function,
                        inference_config,
                        clients,
                        inference_params,
                    )
                    .await
            }
            VariantConfig::BestOfNSampling(params) => {
                params
                    .infer_stream(
                        input,
                        models,
                        function,
                        inference_config,
                        clients,
                        inference_params,
                    )
                    .await
            }
            VariantConfig::Dicl(params) => {
                params
                    .infer_stream(
                        input,
                        models,
                        function,
                        inference_config,
                        clients,
                        inference_params,
                    )
                    .await
            }
            VariantConfig::MixtureOfN(params) => {
                params
                    .infer_stream(
                        input,
                        models,
                        function,
                        inference_config,
                        clients,
                        inference_params,
                    )
                    .await
            }
        }
    }

    #[instrument(skip_all, fields(variant_name = %inference_configs.first().map(|x| x.variant_name.unwrap_or("")).unwrap_or("")))]
    async fn start_batch_inference<'a>(
        &'a self,
        inputs: &[ResolvedInput],
        models: &'a InferenceModels<'a>,
        function: &'a FunctionConfig,
        inference_configs: &'a [InferenceConfig<'a, 'a>],
        clients: &'a InferenceClients<'a>,
        inference_params: Vec<InferenceParams>,
    ) -> Result<StartBatchModelInferenceWithMetadata<'a>, Error> {
        match self {
            VariantConfig::ChatCompletion(params) => {
                params
                    .start_batch_inference(
                        inputs,
                        models,
                        function,
                        inference_configs,
                        clients,
                        inference_params,
                    )
                    .await
            }
            _ => {
                Err(ErrorDetails::UnsupportedVariantForBatchInference { variant_name: None }.into())
            }
        }
    }

    #[instrument(skip_all, fields(variant_name = %variant_name))]
    fn validate(
        &self,
        function: &FunctionConfig,
        models: &mut ModelTable,
        embedding_models: &EmbeddingModelTable,
        templates: &TemplateConfig,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(), Error> {
        match self {
            VariantConfig::ChatCompletion(params) => params.validate(
                function,
                models,
                embedding_models,
                templates,
                function_name,
                variant_name,
            ),
            VariantConfig::BestOfNSampling(params) => params.validate(
                function,
                models,
                embedding_models,
                templates,
                function_name,
                variant_name,
            ),
            VariantConfig::Dicl(params) => params.validate(
                function,
                models,
                embedding_models,
                templates,
                function_name,
                variant_name,
            ),
            VariantConfig::MixtureOfN(params) => params.validate(
                function,
                models,
                embedding_models,
                templates,
                function_name,
                variant_name,
            ),
        }
    }

    fn get_all_template_paths(&self) -> Vec<&PathWithContents> {
        match self {
            VariantConfig::ChatCompletion(params) => params.get_all_template_paths(),
            VariantConfig::BestOfNSampling(params) => params.get_all_template_paths(),
            VariantConfig::Dicl(params) => params.get_all_template_paths(),
            VariantConfig::MixtureOfN(params) => params.get_all_template_paths(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_model_inference_request<'a, 'request>(
    messages: Vec<RequestMessage>,
    system: Option<String>,
    function: &'a FunctionConfig,
    inference_config: &'request InferenceConfig<'a, 'request>,
    stream: bool,
    inference_params: &InferenceParams,
    base_json_mode: Option<JsonMode>,
    extra_body: FullExtraBodyConfig,
) -> Result<ModelInferenceRequest<'request>, Error>
where
    'a: 'request,
{
    let json_mode = inference_params
        .chat_completion
        .json_mode
        .or(base_json_mode);

    Ok(match function {
        FunctionConfig::Chat(_) => {
            ModelInferenceRequest {
                messages,
                system,
                inference_id: inference_config.ids.inference_id,
                tool_config: inference_config.tool_config.map(Cow::Borrowed),
                temperature: inference_params.chat_completion.temperature,
                top_p: inference_params.chat_completion.top_p,
                max_tokens: inference_params.chat_completion.max_tokens,
                presence_penalty: inference_params.chat_completion.presence_penalty,
                frequency_penalty: inference_params.chat_completion.frequency_penalty,
                seed: inference_params.chat_completion.seed,
                stream,
                // In chat mode, we fall back to 'JsonMode::Off' - json mode will only be enabled if
                // explicitly requested in `chat_completion` params.
                json_mode: json_mode.unwrap_or(JsonMode::Off).into(),
                function_type: FunctionType::Chat,
                output_schema: inference_config.dynamic_output_schema.map(|v| &v.value),
                extra_body,
                extra_cache_key: inference_config.extra_cache_key.clone(),
            }
        }
        FunctionConfig::Json(json_config) => {
            let tool_config = match json_mode {
                Some(JsonMode::ImplicitTool) => match inference_config.dynamic_output_schema {
                    Some(schema) => Some(Cow::Owned(create_dynamic_implicit_tool_config(
                        schema.value.clone(),
                    ))),
                    None => Some(Cow::Borrowed(&json_config.implicit_tool_call_config)),
                },
                _ => None,
            };
            let output_schema = match inference_config.dynamic_output_schema {
                Some(schema) => Some(&schema.value),
                None => Some(json_config.output_schema.value),
            };
            ModelInferenceRequest {
                messages,
                system,
                tool_config,
                inference_id: inference_config.ids.inference_id,
                temperature: inference_params.chat_completion.temperature,
                top_p: inference_params.chat_completion.top_p,
                max_tokens: inference_params.chat_completion.max_tokens,
                presence_penalty: inference_params.chat_completion.presence_penalty,
                frequency_penalty: inference_params.chat_completion.frequency_penalty,
                seed: inference_params.chat_completion.seed,
                stream,
                // In json mode, we fall back to 'JsonMode::Strict' if it was unset in both
                // the `chat_completions` params and the variant config.
                json_mode: json_mode.unwrap_or(JsonMode::Strict).into(),
                function_type: FunctionType::Json,
                output_schema,
                extra_body,
                extra_cache_key: inference_config.extra_cache_key.clone(),
            }
        }
    })
}

/// Encapsulates all arguments for the `infer_model_request` function
struct InferModelRequestArgs<'a, 'request> {
    request: ModelInferenceRequest<'request>,
    model_name: Arc<str>,
    model_config: &'a ModelConfig,
    function: &'a FunctionConfig,
    inference_config: &'request InferenceConfig<'a, 'request>,
    clients: &'request InferenceClients<'request>,
    inference_params: InferenceParams,
    retry_config: &'a RetryConfig,
}

/// Refactored `infer_model_request` function accepting a single struct argument
#[instrument(fields(model_name = %args.model_name), skip_all)]
async fn infer_model_request<'a, 'request>(
    args: InferModelRequestArgs<'a, 'request>,
) -> Result<InferenceResult, Error> {
    let model_inference_response = (|| async {
        args.model_config
            .infer(&args.request, args.clients, &args.model_name)
            .await
    })
    .retry(args.retry_config.get_backoff())
    .await?;

    let original_response = model_inference_response.raw_response.clone();
    let model_inference_result =
        ModelInferenceResponseWithMetadata::new(model_inference_response, args.model_name);
    let raw_content = model_inference_result.output.clone();
    let usage = model_inference_result.actual_usage();
    let model_inference_results = vec![model_inference_result];

    args.function
        .prepare_response(
            args.inference_config.ids.inference_id,
            raw_content,
            usage,
            model_inference_results,
            args.inference_config,
            args.inference_params,
            Some(original_response),
        )
        .await
}

#[instrument(fields(model_name = %model_name), skip_all)]
async fn infer_model_request_stream<'request>(
    request: ModelInferenceRequest<'request>,
    model_name: Arc<str>,
    model_config: &ModelConfig,
    function: &FunctionConfig,
    clients: &'request InferenceClients<'request>,
    inference_params: InferenceParams,
    retry_config: RetryConfig,
) -> Result<(InferenceResultStream, ModelUsedInfo), Error> {
    let (
        StreamResponse {
            stream,
            raw_request,
            model_provider_name,
            cached,
        },
        input_messages,
    ) = (|| async {
        model_config
            .infer_stream(&request, clients, &model_name)
            .await
    })
    .retry(retry_config.get_backoff())
    .await?;
    let system = request.system.clone();
    let model_used_info = ModelUsedInfo {
        model_name,
        model_provider_name,
        raw_request,
        inference_params,
        previous_model_inference_results: vec![],
        system,
        input_messages,
        cached,
    };
    let config_type = function.config_type();
    let stream =
        stream.map(move |chunk| chunk.map(|chunk| InferenceResultChunk::new(chunk, config_type)));
    Ok((Box::pin(stream), model_used_info))
}

#[derive(Debug, Deserialize, Copy, Clone)]
pub struct RetryConfig {
    pub num_retries: usize,
    pub max_delay_s: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            num_retries: 0,
            max_delay_s: 10.0,
        }
    }
}

impl RetryConfig {
    pub fn get_backoff(&self) -> backon::ExponentialBuilder {
        ExponentialBuilder::default()
            .with_jitter()
            .with_max_delay(Duration::from_secs_f32(self.max_delay_s))
            .with_max_times(self.num_retries)
    }
}

impl<'a> BatchInferenceConfig<'a> {
    pub fn new(
        templates: &'a TemplateConfig,
        tool_configs: Vec<Option<ToolCallConfig>>,
        dynamic_output_schemas: Vec<Option<DynamicJSONSchema>>,
        function_name: &'a str,
        variant_name: Option<&'a str>,
    ) -> Self {
        Self {
            templates,
            tool_configs,
            dynamic_output_schemas,
            function_name,
            variant_name,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{CacheEnabledMode, CacheOptions};
    use crate::clickhouse::ClickHouseConnectionInfo;
    use crate::endpoints::inference::{ChatCompletionInferenceParams, InferenceCredentials};
    use crate::error::ErrorDetails;
    use crate::function::{FunctionConfigChat, FunctionConfigJson};
    use crate::inference::providers::dummy::{
        DummyProvider, DUMMY_INFER_RESPONSE_CONTENT, DUMMY_INFER_USAGE, DUMMY_JSON_RESPONSE_RAW,
        DUMMY_STREAMING_RESPONSE,
    };
    use crate::inference::types::{
        ContentBlockChunk, ModelInferenceRequestJsonMode, RequestMessage, Role,
    };
    use crate::jsonschema_util::JSONSchemaFromPath;
    use crate::minijinja_util::tests::get_test_template_config;
    use crate::model::{ModelProvider, ProviderConfig};
    use crate::tool::{ToolCallConfig, ToolChoice};
    use reqwest::Client;
    use serde_json::json;
    use std::collections::HashMap;
    use tracing_test::traced_test;

    #[tokio::test]
    async fn test_prepare_model_inference_request() {
        // Setup common variables
        let templates = get_test_template_config();
        let stream = false;

        // Define a dummy tool config for testing
        let tool_config = ToolCallConfig {
            tools_available: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        };

        // Create a sample inference config
        let inference_config = InferenceConfig {
            templates: &templates,
            tool_config: Some(&tool_config),
            function_name: "test_function",
            variant_name: Some("test_variant"),
            dynamic_output_schema: None,
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            extra_body: Default::default(),
            extra_cache_key: None,
        };

        // Define common inference parameters
        let inference_params = InferenceParams {
            chat_completion: ChatCompletionInferenceParams {
                temperature: Some(0.7),
                max_tokens: Some(50),
                top_p: Some(0.9),
                presence_penalty: Some(0.0),
                frequency_penalty: Some(0.0),
                seed: Some(42),
                json_mode: None,
            },
        };

        // Prepare sample messages and system prompt
        let messages = vec![
            RequestMessage {
                role: Role::User,
                content: vec!["Hello, how are you?".to_string().into()],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec!["I'm fine, thank you!".to_string().into()],
            },
        ];
        let system = Some("You are a helpful assistant.".to_string());

        // Test case 1: FunctionConfig::Chat with JsonMode::Off
        let function_config_chat = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        });
        let json_mode = JsonMode::Off;

        let result = prepare_model_inference_request(
            messages.clone(),
            system.clone(),
            &function_config_chat,
            &inference_config,
            stream,
            &inference_params,
            Some(json_mode),
            Default::default(),
        )
        .unwrap();

        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.system, system);
        assert_eq!(result.tool_config, Some(Cow::Borrowed(&tool_config)));
        assert_eq!(result.temperature, Some(0.7));
        assert_eq!(result.top_p, Some(0.9));
        assert_eq!(result.max_tokens, Some(50));
        assert_eq!(result.presence_penalty, Some(0.0));
        assert_eq!(result.frequency_penalty, Some(0.0));
        assert_eq!(result.seed, Some(42));
        assert_eq!(result.stream, stream);
        assert_eq!(result.json_mode, ModelInferenceRequestJsonMode::Off);
        assert_eq!(result.function_type, FunctionType::Chat);
        assert_eq!(result.output_schema, None);

        // Test case 2: FunctionConfig::Json with JsonMode::On and static output schema
        let output_schema_value = json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            },
            "required": ["answer"],
        });
        let output_schema = JSONSchemaFromPath::from_value(&output_schema_value).unwrap();
        let implicit_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema_value);

        let function_config_json = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            assistant_schema: None,
            system_schema: None,
            user_schema: None,
            output_schema: output_schema.clone(),
            implicit_tool_call_config: implicit_tool_call_config.clone(),
        });

        let json_mode = JsonMode::On;

        let result = prepare_model_inference_request(
            messages.clone(),
            system.clone(),
            &function_config_json,
            &inference_config,
            stream,
            &inference_params,
            Some(json_mode),
            Default::default(),
        )
        .unwrap();

        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.system, system.clone());
        assert_eq!(result.tool_config, None);
        assert_eq!(result.temperature, Some(0.7));
        assert_eq!(result.max_tokens, Some(50));
        assert_eq!(result.seed, Some(42));
        assert_eq!(result.stream, stream);
        assert_eq!(result.json_mode, ModelInferenceRequestJsonMode::On);
        assert_eq!(result.function_type, FunctionType::Json);
        assert_eq!(result.output_schema, Some(&output_schema_value));

        // Test case 3: FunctionConfig::Json with JsonMode::ImplicitTool and dynamic output schema
        let dynamic_output_schema_value = json!({
            "type": "object",
            "properties": {
                "result": { "type": "string" }
            },
            "required": ["result"],
        });
        let dynamic_output_schema = DynamicJSONSchema::new(dynamic_output_schema_value.clone());
        let inference_config_dynamic = InferenceConfig {
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            templates: &templates,
            tool_config: Some(&tool_config),
            function_name: "test_function",
            variant_name: Some("test_variant"),
            dynamic_output_schema: Some(&dynamic_output_schema),
            extra_body: Default::default(),
            extra_cache_key: None,
        };
        let json_mode = JsonMode::ImplicitTool;

        let result = prepare_model_inference_request(
            messages.clone(),
            system.clone(),
            &function_config_json,
            &inference_config_dynamic,
            stream,
            &inference_params,
            Some(json_mode),
            Default::default(),
        )
        .unwrap();

        assert_eq!(
            result.tool_config,
            Some(Cow::Owned(create_dynamic_implicit_tool_config(
                dynamic_output_schema_value.clone(),
            )))
        );
        assert_eq!(result.output_schema, Some(&dynamic_output_schema_value));

        // Test case 4: FunctionConfig::Json with JsonMode::Strict
        let json_mode = JsonMode::Strict;

        let result = prepare_model_inference_request(
            messages.clone(),
            system.clone(),
            &function_config_json,
            &inference_config,
            stream,
            &inference_params,
            Some(json_mode),
            Default::default(),
        )
        .unwrap();

        assert_eq!(result.tool_config, None);
        assert_eq!(result.output_schema, Some(&output_schema_value));
        assert_eq!(result.json_mode, ModelInferenceRequestJsonMode::Strict);

        // Test case 5: FunctionConfig::Json with JsonMode::Off (should still set output_schema)
        let json_mode = JsonMode::Off;

        let result = prepare_model_inference_request(
            messages,
            system,
            &function_config_json,
            &inference_config,
            stream,
            &inference_params,
            Some(json_mode),
            Default::default(),
        )
        .unwrap();

        assert_eq!(result.tool_config, None);
        assert_eq!(result.output_schema, Some(&output_schema_value));
        assert_eq!(result.json_mode, ModelInferenceRequestJsonMode::Off);
    }

    #[tokio::test]
    async fn test_infer_model_request() {
        // Setup common variables
        let api_keys = InferenceCredentials::default();
        let client = Client::new();
        let clickhouse_connection_info = ClickHouseConnectionInfo::Disabled;
        let clients = InferenceClients {
            http_client: &client,
            clickhouse_connection_info: &clickhouse_connection_info,
            credentials: &api_keys,
            cache_options: &CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
        };
        let templates = get_test_template_config();
        let inference_params = InferenceParams::default();
        let inference_config = InferenceConfig {
            templates: &templates,
            tool_config: None,
            function_name: "test_function",
            variant_name: Some("test_variant"),
            dynamic_output_schema: None,
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            extra_body: Default::default(),
            extra_cache_key: None,
        };

        // Test case 1: Successful inference with ChatCompletionConfig and FunctionConfigChat
        let model_name = "dummy_chat_model";
        let function_config_chat = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        });

        let request_messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello, how are you?".to_string().into()],
        }];

        let model_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: request_messages.clone(),
            system: None,
            temperature: Some(0.7),
            max_tokens: Some(100),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            output_schema: None,
            tool_config: None,
            function_type: FunctionType::Chat,
            extra_body: Default::default(),
            ..Default::default()
        };

        // Create a dummy provider config with the desired model name
        let dummy_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: model_name.to_string(),
            ..Default::default()
        });

        // Create a model config with the dummy provider
        let model_config = ModelConfig {
            routing: vec![model_name.into()],
            providers: HashMap::from([(
                model_name.into(),
                ModelProvider {
                    name: model_name.into(),
                    config: dummy_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                },
            )]),
        };
        let retry_config = Box::leak(Box::new(RetryConfig::default()));

        // Create the arguments struct
        let args = InferModelRequestArgs {
            request: model_request.clone(),
            model_name: model_name.into(),
            model_config: &model_config,
            function: &function_config_chat,
            inference_config: &inference_config,
            clients: &clients,
            inference_params: inference_params.clone(),
            retry_config,
        };

        // Refactored function call
        let result = infer_model_request(args).await;

        let inference_result = result.unwrap();
        match inference_result {
            InferenceResult::Chat(chat_result) => {
                // The DummyProvider returns DUMMY_INFER_RESPONSE_CONTENT by default
                let expected_content = vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()];
                assert_eq!(chat_result.content, expected_content);
                assert_eq!(chat_result.usage, DUMMY_INFER_USAGE.clone());
                assert_eq!(chat_result.model_inference_results.len(), 1);
                assert_eq!(
                    &*chat_result.model_inference_results[0].model_name,
                    model_name
                );
                // Need to recreate to make this ContentBlock rather than ContentBlockOutput
                let expected_content = vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()];
                assert_eq!(
                    &*chat_result.model_inference_results[0].output,
                    expected_content
                );
            }
            _ => panic!("Expected Chat inference result"),
        }

        // Test case 2: Successful inference with FunctionConfigJson
        let model_name_json = "json";
        let function_config_json = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            output_schema: JSONSchemaFromPath::from_value(&json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string" }
                },
                "required": ["answer"]
            }))
            .unwrap(),
            implicit_tool_call_config: crate::tool::ToolCallConfig {
                tools_available: vec![],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: None,
            },
        });
        let output_schema = json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            },
            "required": ["answer"]
        });

        let model_request_json = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: request_messages.clone(),
            system: None,
            temperature: Some(0.7),
            max_tokens: Some(100),
            seed: None,
            stream: false,
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            json_mode: ModelInferenceRequestJsonMode::On,
            output_schema: Some(&output_schema),
            tool_config: None,
            function_type: FunctionType::Json,
            extra_body: Default::default(),
            ..Default::default()
        };

        // Create a dummy provider config with model_name "json" to trigger JSON response
        let dummy_provider_config_json = ProviderConfig::Dummy(DummyProvider {
            model_name: model_name_json.to_string(),
            ..Default::default()
        });

        let model_config_json = ModelConfig {
            routing: vec![model_name_json.into()],
            providers: HashMap::from([(
                model_name_json.into(),
                ModelProvider {
                    name: model_name_json.into(),
                    config: dummy_provider_config_json,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                },
            )]),
        };

        // Create the arguments struct
        let args = InferModelRequestArgs {
            request: model_request_json.clone(),
            model_name: model_name_json.into(),
            model_config: &model_config_json,
            function: &function_config_json,
            inference_config: &inference_config,
            clients: &clients,
            inference_params: inference_params.clone(),
            retry_config,
        };

        // Refactored function call
        let result = infer_model_request(args).await;

        let inference_result = result.unwrap();
        match inference_result {
            InferenceResult::Json(json_result) => {
                let expected_raw_output = DUMMY_JSON_RESPONSE_RAW.to_string();
                assert_eq!(json_result.output.raw, expected_raw_output);
                assert_eq!(json_result.output.parsed, Some(json!({"answer": "Hello"})));
                assert_eq!(json_result.usage, DUMMY_INFER_USAGE.clone());
                assert_eq!(json_result.model_inference_results.len(), 1);
                assert_eq!(
                    &*json_result.model_inference_results[0].model_name,
                    model_name_json
                );
                assert_eq!(
                    json_result.model_inference_results[0].output,
                    vec![expected_raw_output.into()]
                );
            }
            _ => panic!("Expected Json inference result"),
        }

        // Test case 3: Model inference failure
        let error_model_name = "error";
        let error_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: error_model_name.to_string(),
            ..Default::default()
        });

        let error_model_config = ModelConfig {
            routing: vec![error_model_name.into()],
            providers: HashMap::from([(
                error_model_name.into(),
                ModelProvider {
                    name: error_model_name.into(),
                    config: error_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                },
            )]),
        };

        // Create the arguments struct
        let args = InferModelRequestArgs {
            request: model_request.clone(),
            model_name: error_model_name.into(),
            model_config: &error_model_config,
            function: &function_config_chat,
            inference_config: &inference_config,
            clients: &clients,
            inference_params: inference_params.clone(),
            retry_config,
        };

        // Refactored function call
        let result = infer_model_request(args).await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(matches!(
            error.get_details(),
            ErrorDetails::ModelProvidersExhausted { .. }
        ));
    }

    #[tokio::test]
    #[traced_test]
    async fn test_infer_model_request_errors() {
        // Setup common variables
        let api_keys = InferenceCredentials::default();
        let client = Client::new();
        let clickhouse_connection_info = ClickHouseConnectionInfo::Disabled;
        let clients = InferenceClients {
            http_client: &client,
            clickhouse_connection_info: &clickhouse_connection_info,
            credentials: &api_keys,
            cache_options: &CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
        };
        let templates = get_test_template_config();
        let inference_params = InferenceParams::default();
        let inference_config = InferenceConfig {
            templates: &templates,
            tool_config: None,
            function_name: "test_function",
            variant_name: Some("test_variant"),
            dynamic_output_schema: None,
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            extra_body: Default::default(),
            extra_cache_key: None,
        };

        let model_name = "dummy_chat_model";
        let error_model_name = "error";
        let function_config_chat = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        });

        let request_messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello, how are you?".to_string().into()],
        }];

        let model_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: request_messages.clone(),
            system: None,
            temperature: Some(0.7),
            max_tokens: Some(100),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            output_schema: None,
            tool_config: None,
            function_type: FunctionType::Chat,
            extra_body: Default::default(),
            ..Default::default()
        };

        // Create a dummy provider config with the error model name
        let error_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: error_model_name.to_string(),
            ..Default::default()
        });

        // Create a dummy provider config with the good model name
        let dummy_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: model_name.to_string(),
            ..Default::default()
        });

        // Create a model config with the dummy provider
        let model_config = ModelConfig {
            routing: vec![error_model_name.into(), model_name.into()],
            providers: HashMap::from([
                (
                    error_model_name.into(),
                    ModelProvider {
                        name: error_model_name.into(),
                        config: error_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                    },
                ),
                (
                    model_name.into(),
                    ModelProvider {
                        name: model_name.into(),
                        config: dummy_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                    },
                ),
            ]),
        };
        let retry_config = Box::leak(Box::new(RetryConfig::default()));

        // Create the arguments struct
        let args = InferModelRequestArgs {
            request: model_request.clone(),
            model_name: model_name.into(),
            model_config: &model_config,
            function: &function_config_chat,
            inference_config: &inference_config,
            clients: &clients,
            inference_params: inference_params.clone(),
            retry_config,
        };

        // Refactored function call
        let result = infer_model_request(args).await;

        let inference_result = result.unwrap();
        match inference_result {
            InferenceResult::Chat(chat_result) => {
                // The DummyProvider returns DUMMY_INFER_RESPONSE_CONTENT by default
                let expected_content = vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()];
                assert_eq!(chat_result.content, expected_content);
                assert_eq!(chat_result.usage, DUMMY_INFER_USAGE.clone());
                assert_eq!(chat_result.model_inference_results.len(), 1);
                assert_eq!(
                    &*chat_result.model_inference_results[0].model_name,
                    model_name
                );
                // Need to recreate to make this ContentBlock rather than ContentBlockOutput
                let expected_content = vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()];
                assert_eq!(
                    chat_result.model_inference_results[0].output,
                    expected_content
                );
            }
            _ => panic!("Expected Chat inference result"),
        }
        assert!(logs_contain(
            r#"ERROR test_infer_model_request_errors:infer_model_request{model_name=dummy_chat_model}:infer{provider_name="error"}: tensorzero_internal::error: Error from dummy client: Error sending request to Dummy provider for model 'error'."#
        ));
    }

    #[tokio::test]
    async fn test_infer_model_request_stream() {
        // Set up the HTTP client and ClickHouse connection info
        let client = reqwest::Client::new();
        let clickhouse_connection_info = ClickHouseConnectionInfo::Disabled;
        let api_keys = InferenceCredentials::default();
        let clients = InferenceClients {
            http_client: &client,
            clickhouse_connection_info: &clickhouse_connection_info,
            credentials: &api_keys,
            cache_options: &CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
        };
        let retry_config = RetryConfig::default();
        // Create a dummy function config (chat completion)
        let function_config = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            tools: vec![],
            tool_choice: crate::tool::ToolChoice::Auto,
            parallel_tool_calls: None,
        });

        // Create an input message
        let messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello, how are you?".to_string().into()],
        }];
        let system = Some("You are a helpful assistant.".to_string());

        // Create a dummy model config with a provider
        let dummy_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            ..Default::default()
        });

        let model_config = Box::leak(Box::new(ModelConfig {
            routing: vec!["good_provider".into()],
            providers: HashMap::from([(
                "good_provider".into(),
                ModelProvider {
                    name: "good_provider".into(),
                    config: dummy_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                },
            )]),
        }));

        // Prepare the model inference request
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages,
            system,
            temperature: Some(0.7),
            max_tokens: Some(50),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::Off,
            output_schema: None,
            seed: None,
            tool_config: None,
            function_type: FunctionType::Chat,
            extra_body: Default::default(),
            ..Default::default()
        };

        // Initialize inference parameters
        let inference_params = InferenceParams::default();

        // Call infer_model_request_stream
        let result = infer_model_request_stream(
            request,
            "good_model".into(),
            model_config,
            &function_config,
            &clients,
            inference_params.clone(),
            retry_config,
        )
        .await;

        // Assert that the result is OK
        assert!(result.is_ok());

        // Unwrap the result
        let (mut stream, model_used_info) = result.unwrap();

        // Check the first chunk
        if let InferenceResultChunk::Chat(chat_chunk) = stream.next().await.unwrap().unwrap() {
            assert_eq!(chat_chunk.content.len(), 1);
            if let ContentBlockChunk::Text(text_chunk) = &chat_chunk.content[0] {
                assert_eq!(text_chunk.text, DUMMY_STREAMING_RESPONSE[0]);
            } else {
                panic!("Expected text chunk in first inference result chunk.");
            }
        } else {
            panic!("Expected chat inference result chunk.");
        }

        // Verify the model used information
        assert_eq!(&*model_used_info.model_name, "good_model");
        assert_eq!(&*model_used_info.model_provider_name, "good_provider");
        assert_eq!(model_used_info.inference_params, inference_params);

        // Iterate over the stream and collect the remaining chunks
        let mut received_text = String::new();
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.expect("Stream chunk should be OK.");

            if let InferenceResultChunk::Chat(chat_chunk) = chunk {
                for content_block in chat_chunk.content {
                    if let ContentBlockChunk::Text(text_chunk) = content_block {
                        received_text.push_str(&text_chunk.text);
                    }
                }
            } else if let Some(usage) = chunk.usage() {
                // Verify the usage information
                assert_eq!(usage.input_tokens, 10);
                assert_eq!(usage.output_tokens, DUMMY_STREAMING_RESPONSE.len() as u32);
            } else {
                panic!("Unexpected inference result chunk.");
            }
        }

        // Combine the first chunk's text with the received text
        let mut full_response = DUMMY_STREAMING_RESPONSE[0].to_string();
        full_response.push_str(&received_text);

        // Verify the full response
        let expected_response: String = DUMMY_STREAMING_RESPONSE.iter().cloned().collect();
        assert_eq!(full_response, expected_response);
    }

    #[tokio::test]
    #[traced_test]
    async fn test_infer_model_request_errors_stream() {
        // Setup common variables
        let api_keys = InferenceCredentials::default();
        let client = Client::new();
        let clickhouse_connection_info = ClickHouseConnectionInfo::Disabled;
        let clients = InferenceClients {
            http_client: &client,
            clickhouse_connection_info: &clickhouse_connection_info,
            credentials: &api_keys,
            cache_options: &CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
        };
        let inference_params = InferenceParams::default();

        let model_name = "dummy_chat_model";
        let error_model_name = "error";
        let function_config_chat = Box::leak(Box::new(FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        })));

        let request_messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello, how are you?".to_string().into()],
        }];

        let model_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: request_messages.clone(),
            system: None,
            temperature: Some(0.7),
            max_tokens: Some(100),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            output_schema: None,
            tool_config: None,
            function_type: FunctionType::Chat,
            extra_body: Default::default(),
            ..Default::default()
        };

        // Create a dummy provider config with the error model name
        let error_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: error_model_name.to_string(),
            ..Default::default()
        });

        // Create a dummy provider config with the good model name
        let dummy_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: model_name.to_string(),
            ..Default::default()
        });

        // Create a model config with the dummy provider
        let model_config = Box::leak(Box::new(ModelConfig {
            routing: vec![error_model_name.into(), model_name.into()],
            providers: HashMap::from([
                (
                    error_model_name.into(),
                    ModelProvider {
                        name: error_model_name.into(),
                        config: error_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                    },
                ),
                (
                    model_name.into(),
                    ModelProvider {
                        name: model_name.into(),
                        config: dummy_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                    },
                ),
            ]),
        }));
        let retry_config = RetryConfig::default();

        // Call infer_model_request_stream
        let result = infer_model_request_stream(
            model_request,
            model_name.into(),
            model_config,
            function_config_chat,
            &clients,
            inference_params.clone(),
            retry_config,
        )
        .await;

        // Assert that the result is OK
        assert!(result.is_ok());

        // Unwrap the result
        let (mut stream, model_used_info) = result.unwrap();

        // Check the first chunk
        if let InferenceResultChunk::Chat(chat_chunk) = stream.next().await.unwrap().unwrap() {
            assert_eq!(chat_chunk.content.len(), 1);
            if let ContentBlockChunk::Text(text_chunk) = &chat_chunk.content[0] {
                assert_eq!(text_chunk.text, DUMMY_STREAMING_RESPONSE[0]);
            } else {
                panic!("Expected text chunk in first inference result chunk.");
            }
        } else {
            panic!("Expected chat inference result chunk.");
        }

        // Verify the model used information
        assert_eq!(&*model_used_info.model_name, model_name);
        assert_eq!(&*model_used_info.model_provider_name, model_name);
        assert_eq!(model_used_info.inference_params, inference_params);

        // Iterate over the stream and collect the remaining chunks
        let mut received_text = String::new();
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.expect("Stream chunk should be OK.");

            if let InferenceResultChunk::Chat(chat_chunk) = chunk {
                for content_block in chat_chunk.content {
                    if let ContentBlockChunk::Text(text_chunk) = content_block {
                        received_text.push_str(&text_chunk.text);
                    }
                }
            } else if let Some(usage) = chunk.usage() {
                // Verify the usage information
                assert_eq!(usage.input_tokens, 10);
                assert_eq!(usage.output_tokens, DUMMY_STREAMING_RESPONSE.len() as u32);
            } else {
                panic!("Unexpected inference result chunk.");
            }
        }

        // Combine the first chunk's text with the received text
        let mut full_response = DUMMY_STREAMING_RESPONSE[0].to_string();
        full_response.push_str(&received_text);

        // Verify the full response
        let expected_response: String = DUMMY_STREAMING_RESPONSE.iter().cloned().collect();
        assert_eq!(full_response, expected_response);

        assert!(logs_contain(
            r#"ERROR test_infer_model_request_errors_stream:infer_model_request_stream{model_name=dummy_chat_model}:infer_stream{provider_name="error"}: tensorzero_internal::error: Error from dummy client: Error sending request to Dummy provider for model 'error'."#
        ));
    }
}
