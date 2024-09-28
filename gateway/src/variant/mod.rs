use futures::StreamExt;
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

use crate::embeddings::EmbeddingModelConfig;
use crate::endpoints::inference::{InferenceClients, InferenceModels, InferenceParams};
use crate::error::Error;
use crate::function::FunctionConfig;
use crate::inference::types::{
    FunctionType, InferenceResultChunk, InferenceResultStream, Input, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, ModelInferenceResponseWithMetadata, RequestMessage,
};
use crate::jsonschema_util::DynamicJSONSchema;
use crate::minijinja_util::TemplateConfig;
use crate::tool::{create_dynamic_implicit_tool_config, ToolCallConfig};
use crate::{inference::types::InferenceResult, model::ModelConfig};
pub mod best_of_n;
pub mod chat_completion;
pub mod dicl;

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum VariantConfig {
    ChatCompletion(chat_completion::ChatCompletionConfig),
    #[serde(rename = "experimental_best_of_n")]
    BestOfN(best_of_n::BestOfNConfig),
    #[serde(rename = "experimental_dynamic_in_context_learning")]
    Dicl(dicl::DiclConfig),
}

/// This type is used to determine how to enforce JSON mode for a given variant.
/// Variants represent JSON mode in a slightly more abstract sense than ModelInferenceRequests, as
/// we support coercing tool calls into JSON mode.
/// This is represented as a tool config in the
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum JsonMode {
    Off,
    #[default]
    On,
    Strict,
    ImplicitTool,
}

/// Maps to the subset of Config that applies to the current inference request.
/// It doesn't take into account inference-time overrides (e.g. dynamic tools).
pub struct InferenceConfig<'a> {
    pub tool_config: Option<ToolCallConfig>,
    pub templates: &'a TemplateConfig<'a>,
    pub dynamic_output_schema: Option<DynamicJSONSchema>,
    pub function_name: String,
    pub variant_name: String,
}

pub struct ModelUsedInfo<'a> {
    pub model_name: &'a str,
    pub model_provider_name: &'a str,
    pub raw_request: String,
    pub inference_params: InferenceParams,
    pub previous_model_inference_results: Vec<ModelInferenceResponseWithMetadata<'a>>,
}

pub trait Variant {
    async fn infer<'a, 'request>(
        &'a self,
        input: &Input,
        models: &'request InferenceModels<'a>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        clients: &'request InferenceClients,
        inference_params: InferenceParams,
    ) -> Result<InferenceResult<'a>, Error>;

    async fn infer_stream<'request>(
        &'static self,
        input: &Input,
        models: &'request InferenceModels<'static>,
        function: &'static FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        clients: &'request InferenceClients<'request>,
        inference_params: InferenceParams,
    ) -> Result<
        (
            InferenceResultChunk,
            InferenceResultStream,
            ModelUsedInfo<'static>,
        ),
        Error,
    >;

    fn validate(
        &self,
        function: &FunctionConfig,
        models: &HashMap<String, ModelConfig>,
        embedding_models: &HashMap<String, EmbeddingModelConfig>,
        templates: &TemplateConfig,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(), Error>;

    fn get_all_template_paths(&self) -> Vec<&PathBuf>;
}

impl VariantConfig {
    pub fn weight(&self) -> f64 {
        match self {
            VariantConfig::ChatCompletion(params) => params.weight,
            VariantConfig::BestOfN(params) => params.weight,
            VariantConfig::Dicl(params) => params.weight,
        }
    }
}

impl Variant for VariantConfig {
    async fn infer<'a, 'request>(
        &'a self,
        input: &Input,
        models: &'request InferenceModels<'a>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        clients: &'request InferenceClients<'request>,
        inference_params: InferenceParams,
    ) -> Result<InferenceResult<'a>, Error> {
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
            VariantConfig::BestOfN(params) => {
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
        }
    }

    async fn infer_stream<'request>(
        &'static self,
        input: &Input,
        models: &'request InferenceModels<'static>,
        function: &'static FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        clients: &'request InferenceClients<'request>,
        inference_params: InferenceParams,
    ) -> Result<
        (
            InferenceResultChunk,
            InferenceResultStream,
            ModelUsedInfo<'static>,
        ),
        Error,
    > {
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
            VariantConfig::BestOfN(params) => {
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
        }
    }

    fn validate(
        &self,
        function: &FunctionConfig,
        models: &HashMap<String, ModelConfig>,
        embedding_models: &HashMap<String, EmbeddingModelConfig>,
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
            VariantConfig::BestOfN(params) => params.validate(
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
        }
    }

    fn get_all_template_paths(&self) -> Vec<&PathBuf> {
        match self {
            VariantConfig::ChatCompletion(params) => params.get_all_template_paths(),
            VariantConfig::BestOfN(params) => params.get_all_template_paths(),
            VariantConfig::Dicl(params) => params.get_all_template_paths(),
        }
    }
}

fn prepare_model_inference_request<'a, 'request>(
    messages: Vec<RequestMessage>,
    system: Option<String>,
    function: &'a FunctionConfig,
    inference_config: &'request InferenceConfig<'request>,
    stream: bool,
    inference_params: &InferenceParams,
    json_mode: &'a JsonMode,
) -> Result<ModelInferenceRequest<'request>, Error>
where
    'a: 'request,
{
    Ok(match function {
        FunctionConfig::Chat(_) => ModelInferenceRequest {
            messages,
            system,
            tool_config: inference_config.tool_config.as_ref().map(Cow::Borrowed),
            temperature: inference_params.chat_completion.temperature,
            max_tokens: inference_params.chat_completion.max_tokens,
            seed: inference_params.chat_completion.seed,
            stream,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        },
        FunctionConfig::Json(json_config) => {
            let tool_config = match json_mode {
                JsonMode::ImplicitTool => match &inference_config.dynamic_output_schema {
                    Some(schema) => Some(Cow::Owned(create_dynamic_implicit_tool_config(
                        schema.value.clone(),
                    ))),
                    None => Some(Cow::Borrowed(&json_config.implicit_tool_call_config)),
                },
                _ => None,
            };
            let output_schema = match &inference_config.dynamic_output_schema {
                Some(schema) => Some(&schema.value),
                None => Some(json_config.output_schema.value),
            };
            ModelInferenceRequest {
                messages,
                system,
                tool_config,
                temperature: inference_params.chat_completion.temperature,
                max_tokens: inference_params.chat_completion.max_tokens,
                seed: inference_params.chat_completion.seed,
                stream,
                json_mode: json_mode.into(),
                function_type: FunctionType::Json,
                output_schema,
            }
        }
    })
}

/// If you are in a variant and you want to make an inference request to a model,
/// use this function unless you are calling another variant's `infer` function
///
/// Takes a ModelInferenceRequest and makes an inference request to the model.
/// The model_name is only for bookkeeping and is not used by the model itself.
async fn infer_model_request<'a, 'request>(
    request: ModelInferenceRequest<'request>,
    model_name: &'a str,
    model_config: &'a ModelConfig,
    function: &'a FunctionConfig,
    inference_config: &'request InferenceConfig<'request>,
    clients: &'request InferenceClients<'request>,
    inference_params: InferenceParams,
) -> Result<InferenceResult<'a>, Error> {
    let model_inference_response = model_config.infer(&request, clients.http_client).await?;
    let model_inference_result =
        ModelInferenceResponseWithMetadata::new(model_inference_response, model_name);
    let inference_id = Uuid::now_v7();
    let raw_content = model_inference_result.content.clone();
    let usage = model_inference_result.usage.clone();
    let model_inference_results = vec![model_inference_result];
    function
        .prepare_response(
            inference_id,
            raw_content,
            usage,
            model_inference_results,
            inference_config,
            inference_params,
        )
        .await
}

async fn infer_model_request_stream<'request>(
    request: ModelInferenceRequest<'request>,
    model_name: &'static str,
    model_config: &'static ModelConfig,
    function: &'static FunctionConfig,
    clients: &'request InferenceClients<'request>,
    inference_params: InferenceParams,
) -> Result<
    (
        InferenceResultChunk,
        InferenceResultStream,
        ModelUsedInfo<'static>,
    ),
    Error,
> {
    let (first_chunk, stream, raw_request, model_provider_name) = model_config
        .infer_stream(&request, clients.http_client)
        .await?;
    let model_used_info = ModelUsedInfo {
        model_name,
        model_provider_name,
        raw_request,
        inference_params,
        previous_model_inference_results: vec![],
    };
    let first_chunk = InferenceResultChunk::new(first_chunk, function);
    let stream =
        stream.map(move |chunk| chunk.map(|chunk| InferenceResultChunk::new(chunk, function)));
    Ok((first_chunk, Box::pin(stream), model_used_info))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clickhouse::ClickHouseConnectionInfo;
    use crate::endpoints::inference::ChatCompletionInferenceParams;
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
    use crate::model::ProviderConfig;
    use crate::tool::{ToolCallConfig, ToolChoice};
    use reqwest::Client;
    use serde_json::json;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_prepare_model_inference_request() {
        // Setup common variables
        let templates = get_test_template_config();
        let stream = false;

        // Define a dummy tool config for testing
        let tool_config = ToolCallConfig {
            tools_available: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: false,
        };

        // Create a sample inference config
        let inference_config = InferenceConfig {
            templates: &templates,
            tool_config: Some(tool_config.clone()),
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            dynamic_output_schema: None,
        };

        // Define common inference parameters
        let inference_params = InferenceParams {
            chat_completion: ChatCompletionInferenceParams {
                temperature: Some(0.7),
                max_tokens: Some(50),
                seed: Some(42),
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
            parallel_tool_calls: false,
        });
        let json_mode = JsonMode::Off;

        let result = prepare_model_inference_request(
            messages.clone(),
            system.clone(),
            &function_config_chat,
            &inference_config,
            stream,
            &inference_params,
            &json_mode,
        )
        .unwrap();

        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.system, system);
        assert_eq!(result.tool_config, Some(Cow::Borrowed(&tool_config)));
        assert_eq!(result.temperature, Some(0.7));
        assert_eq!(result.max_tokens, Some(50));
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
            &json_mode,
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
            templates: &templates,
            tool_config: Some(tool_config.clone()),
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            dynamic_output_schema: Some(dynamic_output_schema.clone()),
        };

        let json_mode = JsonMode::ImplicitTool;

        let result = prepare_model_inference_request(
            messages.clone(),
            system.clone(),
            &function_config_json,
            &inference_config_dynamic,
            stream,
            &inference_params,
            &json_mode,
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
            &json_mode,
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
            &json_mode,
        )
        .unwrap();

        assert_eq!(result.tool_config, None);
        assert_eq!(result.output_schema, Some(&output_schema_value));
        assert_eq!(result.json_mode, ModelInferenceRequestJsonMode::Off);
    }

    #[tokio::test]
    async fn test_infer_model_request() {
        // Setup common variables
        let client = Client::new();
        let clickhouse_connection_info = ClickHouseConnectionInfo::Disabled;
        let clients = InferenceClients {
            http_client: &client,
            clickhouse_connection_info: &clickhouse_connection_info,
        };
        let templates = get_test_template_config();
        let inference_params = InferenceParams::default();
        let inference_config = InferenceConfig {
            templates: &templates,
            tool_config: None,
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            dynamic_output_schema: None,
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
            parallel_tool_calls: false,
        });

        let request_messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello, how are you?".to_string().into()],
        }];

        let model_request = ModelInferenceRequest {
            messages: request_messages.clone(),
            system: None,
            temperature: Some(0.7),
            max_tokens: Some(100),
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            output_schema: None,
            tool_config: None,
            function_type: FunctionType::Chat,
        };

        // Create a dummy provider config with the desired model name
        let dummy_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: model_name.to_string(),
        });

        // Create a model config with the dummy provider
        let model_config = ModelConfig {
            routing: vec![model_name.to_string()],
            providers: HashMap::from([(model_name.to_string(), dummy_provider_config)]),
        };

        let result = infer_model_request(
            model_request.clone(),
            model_name,
            &model_config,
            &function_config_chat,
            &inference_config,
            &clients,
            inference_params.clone(),
        )
        .await;

        let inference_result = result.unwrap();
        match inference_result {
            InferenceResult::Chat(chat_result) => {
                // The DummyProvider returns DUMMY_INFER_RESPONSE_CONTENT by default
                let expected_content = vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()];
                assert_eq!(chat_result.content, expected_content);
                assert_eq!(chat_result.usage, DUMMY_INFER_USAGE.clone());
                assert_eq!(chat_result.model_inference_results.len(), 1);
                assert_eq!(
                    chat_result.model_inference_results[0].model_name,
                    model_name
                );
                // Need to recreate to make this ContentBlock rather than ContentBlockOutput
                let expected_content = vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()];
                assert_eq!(
                    chat_result.model_inference_results[0].content,
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
                parallel_tool_calls: false,
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
            messages: request_messages.clone(),
            system: None,
            temperature: Some(0.7),
            max_tokens: Some(100),
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            output_schema: Some(&output_schema),
            tool_config: None,
            function_type: FunctionType::Json,
        };

        // Create a dummy provider config with model_name "json" to trigger JSON response
        let dummy_provider_config_json = ProviderConfig::Dummy(DummyProvider {
            model_name: model_name_json.to_string(),
        });

        let model_config_json = ModelConfig {
            routing: vec![model_name_json.to_string()],
            providers: HashMap::from([(model_name_json.to_string(), dummy_provider_config_json)]),
        };

        let result = infer_model_request(
            model_request_json.clone(),
            model_name_json,
            &model_config_json,
            &function_config_json,
            &inference_config,
            &clients,
            inference_params.clone(),
        )
        .await;

        let inference_result = result.unwrap();
        match inference_result {
            InferenceResult::Json(json_result) => {
                let expected_raw_output = DUMMY_JSON_RESPONSE_RAW.to_string();
                assert_eq!(json_result.output.raw, expected_raw_output);
                assert_eq!(json_result.output.parsed, Some(json!({"answer": "Hello"})));
                assert_eq!(json_result.usage, DUMMY_INFER_USAGE.clone());
                assert_eq!(json_result.model_inference_results.len(), 1);
                assert_eq!(
                    json_result.model_inference_results[0].model_name,
                    model_name_json
                );
                assert_eq!(
                    json_result.model_inference_results[0].content,
                    vec![expected_raw_output.into()]
                );
            }
            _ => panic!("Expected Json inference result"),
        }

        // Test case 3: Model inference failure
        let error_model_name = "error";
        let error_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: error_model_name.to_string(),
        });

        let error_model_config = ModelConfig {
            routing: vec![error_model_name.to_string()],
            providers: HashMap::from([(error_model_name.to_string(), error_provider_config)]),
        };

        let result = infer_model_request(
            model_request.clone(),
            error_model_name,
            &error_model_config,
            &function_config_chat,
            &inference_config,
            &clients,
            inference_params.clone(),
        )
        .await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(matches!(error, Error::ModelProvidersExhausted { .. }));
    }

    #[tokio::test]
    async fn test_infer_model_request_stream() {
        // Set up the HTTP client and ClickHouse connection info
        let client = reqwest::Client::new();
        let clickhouse_connection_info = ClickHouseConnectionInfo::Disabled;
        let clients = InferenceClients {
            http_client: &client,
            clickhouse_connection_info: &clickhouse_connection_info,
        };

        // Create a dummy function config (chat completion)
        let function_config = Box::leak(Box::new(FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            tools: vec![],
            tool_choice: crate::tool::ToolChoice::Auto,
            parallel_tool_calls: false,
        })));

        // Create an input message
        let messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello, how are you?".to_string().into()],
        }];
        let system = Some("You are a helpful assistant.".to_string());

        // Create a dummy model config with a provider
        let dummy_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".to_string(),
        });

        let model_config = Box::leak(Box::new(ModelConfig {
            routing: vec!["good_provider".to_string()],
            providers: HashMap::from([("good_provider".to_string(), dummy_provider_config)]),
        }));

        // Prepare the model inference request
        let request = ModelInferenceRequest {
            messages,
            system,
            temperature: Some(0.7),
            max_tokens: Some(50),
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::Off,
            output_schema: None,
            seed: None,
            tool_config: None,
            function_type: FunctionType::Chat,
        };

        // Initialize inference parameters
        let inference_params = InferenceParams::default();

        // Call infer_model_request_stream
        let result = infer_model_request_stream(
            request,
            "good_model",
            model_config,
            function_config,
            &clients,
            inference_params.clone(),
        )
        .await;

        // Assert that the result is OK
        assert!(result.is_ok());

        // Unwrap the result
        let (first_chunk, mut stream, model_used_info) = result.unwrap();

        // Check the first chunk
        if let InferenceResultChunk::Chat(chat_chunk) = first_chunk {
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
        assert_eq!(model_used_info.model_name, "good_model");
        assert_eq!(model_used_info.model_provider_name, "good_provider");
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
}
