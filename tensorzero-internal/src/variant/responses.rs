use std::sync::Arc;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config_parser::{LoadableConfig, PathWithContents};
use crate::embeddings::EmbeddingModelTable;
use crate::endpoints::inference::{InferenceClients, InferenceModels, InferenceParams};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::batch::StartBatchModelInferenceWithMetadata;
use crate::inference::types::{InferenceResult, InferenceResultStream, ResolvedInput};
use crate::minijinja_util::TemplateConfig;
use crate::model::ModelTable;
use crate::variant::{InferenceConfig, JsonMode, ModelUsedInfo, RetryConfig, Variant};

use super::{
    infer_model_request, infer_model_request_stream, prepare_model_inference_request,
    InferModelRequestArgs, 
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResponsesConfig {
    pub model: Arc<str>,
    pub weight: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_mode: Option<JsonMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_template: Option<PathWithContents>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_template: Option<PathWithContents>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assistant_template: Option<PathWithContents>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retries: Option<RetryConfig>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub extra_body: Vec<crate::inference::types::extra_body::ExtraBodyConfig>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub extra_headers: Vec<crate::inference::types::extra_headers::ExtraHeadersConfig>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct UninitializedResponsesConfig {
    pub model: Arc<str>,
    pub weight: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_mode: Option<JsonMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_template: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_template: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assistant_template: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retries: Option<RetryConfig>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub extra_body: Vec<crate::inference::types::extra_body::ExtraBodyConfig>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub extra_headers: Vec<crate::inference::types::extra_headers::ExtraHeadersConfig>,
}

impl LoadableConfig<ResponsesConfig> for UninitializedResponsesConfig {
    fn load<P: AsRef<Path>>(self, base_path: P) -> Result<ResponsesConfig, Error> {
        Ok(ResponsesConfig {
            model: self.model,
            weight: self.weight,
            json_mode: self.json_mode,
            system_template: self
                .system_template
                .map(|path| PathWithContents::from_path(path, Some(&base_path)))
                .transpose()?,
            user_template: self
                .user_template
                .map(|path| PathWithContents::from_path(path, Some(&base_path)))
                .transpose()?,
            assistant_template: self
                .assistant_template
                .map(|path| PathWithContents::from_path(path, Some(&base_path)))
                .transpose()?,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            seed: self.seed,
            top_p: self.top_p,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            retries: self.retries,
            extra_body: self.extra_body,
            extra_headers: self.extra_headers,
        })
    }
}

impl ResponsesConfig {
    fn prepare_system_message(
        &self,
        templates: &crate::minijinja_util::TemplateConfig<'_>,
        system: Option<&serde_json::Value>,
    ) -> Result<Option<String>, Error> {
        use crate::variant::chat_completion::prepare_system_message;
        let template_path = self
            .system_template
            .as_ref()
            .map(|x| {
                x.path
                    .to_str()
                    .ok_or_else(|| Error::new(ErrorDetails::InvalidTemplatePath))
            })
            .transpose()?;
        prepare_system_message(system, templates, template_path)
    }

    fn prepare_request<'a, 'request>(
        &'a self,
        input: &ResolvedInput,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'a, 'request>,
        stream: bool,
        inference_params: &mut InferenceParams,
    ) -> Result<crate::inference::types::ModelInferenceRequest<'request>, Error>
    where
        'a: 'request,
    {
        let messages = input
            .messages
            .iter()
            .map(|message| self.prepare_request_message(inference_config.templates, message))
            .collect::<Result<Vec<_>, _>>()?;
        let system =
            self.prepare_system_message(inference_config.templates, input.system.as_ref())?;

        inference_params
            .chat_completion
            .backfill_with_variant_params(
                self.temperature,
                self.max_tokens,
                self.seed,
                self.top_p,
                self.presence_penalty,
                self.frequency_penalty,
            );

        let extra_body = crate::inference::types::extra_body::FullExtraBodyConfig {
            extra_body: self.extra_body.first().cloned(),
            inference_extra_body: inference_config
                .extra_body
                .clone()
                .filter(inference_config.variant_name),
        };

        let extra_headers = crate::inference::types::extra_headers::FullExtraHeadersConfig {
            variant_extra_headers: self.extra_headers.first().cloned(),
            inference_extra_headers: inference_config
                .extra_headers
                .clone()
                .filter(inference_config.variant_name),
        };

        prepare_model_inference_request(
            messages,
            system,
            function,
            inference_config,
            stream,
            inference_params,
            self.json_mode,
            crate::inference::types::ApiType::Responses, // This is the key difference!
            extra_body,
            extra_headers,
        )
    }

    fn prepare_request_message(
        &self,
        templates: &crate::minijinja_util::TemplateConfig<'_>,
        message: &crate::inference::types::ResolvedInputMessage,
    ) -> Result<crate::inference::types::RequestMessage, Error> {
        use crate::variant::chat_completion::prepare_request_message;
        use crate::inference::types::Role;
        let template_path = match message.role {
            Role::User => self.user_template.as_ref(),
            Role::Assistant => self.assistant_template.as_ref(),
        }
        .map(|x| {
            x.path
                .to_str()
                .ok_or_else(|| Error::new(ErrorDetails::InvalidTemplatePath))
        })
        .transpose()?;
        prepare_request_message(message, templates, template_path)
    }
}
impl Variant for ResponsesConfig {
    async fn infer<'a: 'request, 'request>(
        &self,
        input: &ResolvedInput,
        models: &'request InferenceModels<'a>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'static, 'request>,
        clients: &'request InferenceClients<'request>,
        mut inference_params: InferenceParams,
    ) -> Result<InferenceResult, Error> {
        let request = self.prepare_request(
            input,
            function,
            inference_config,
            false,
            &mut inference_params,
        )?;
        let model_config = models.models.get(&self.model).await?.ok_or_else(|| {
            Error::new(ErrorDetails::UnknownModel {
                name: self.model.to_string(),
            })
        })?;

        let args = InferModelRequestArgs {
            request,
            model_name: self.model.clone(),
            model_config: &model_config,
            function,
            inference_config,
            clients,
            inference_params,
            retry_config: &self.retries.unwrap_or_default(),
        };
        infer_model_request(args).await
    }

    async fn infer_stream<'a, 'request>(
        &self,
        input: &ResolvedInput,
        models: &'request InferenceModels<'a>,
        function: &FunctionConfig,
        inference_config: &'request InferenceConfig<'static, 'request>,
        clients: &'request InferenceClients<'request>,
        mut inference_params: InferenceParams,
    ) -> Result<(InferenceResultStream, ModelUsedInfo), Error> {
        let request = self.prepare_request(
            input,
            function,
            inference_config,
            true,
            &mut inference_params,
        )?;
        let model_config = models.models.get(&self.model).await?.ok_or_else(|| {
            Error::new(ErrorDetails::UnknownModel {
                name: self.model.to_string(),
            })
        })?;
        
        infer_model_request_stream(
            request,
            self.model.clone(),
            &model_config,
            function,
            clients,
            inference_params,
            self.retries.unwrap_or_default(),
        ).await
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _inputs: &[ResolvedInput],
        _models: &'a InferenceModels<'a>,
        _function: &'a FunctionConfig,
        _inference_configs: &'a [InferenceConfig<'a, 'a>],
        _clients: &'a InferenceClients<'a>,
        _inference_params: Vec<InferenceParams>,
    ) -> Result<StartBatchModelInferenceWithMetadata<'a>, Error> {
        // Responses API doesn't support batch inference yet
        Err(ErrorDetails::UnsupportedVariantForBatchInference { 
            variant_name: Some("responses".to_string()) 
        }.into())
    }

    async fn validate(
        &self,
        function: &FunctionConfig,
        models: &mut ModelTable,
        embedding_models: &EmbeddingModelTable,
        templates: &TemplateConfig<'_>,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(), Error> {
        // Reuse chat completion validation logic since they're similar
        let chat_config = crate::variant::chat_completion::ChatCompletionConfig {
            weight: Some(self.weight),
            model: self.model.clone(),
            system_template: self.system_template.clone(),
            user_template: self.user_template.clone(),
            assistant_template: self.assistant_template.clone(),
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed: self.seed,
            json_mode: self.json_mode,
            retries: self.retries.clone().unwrap_or_default(),
            extra_body: self.extra_body.first().cloned(),
            extra_headers: self.extra_headers.first().cloned(),
        };
        
        chat_config.validate(function, models, embedding_models, templates, function_name, variant_name).await
    }

    fn get_all_template_paths(&self) -> Vec<&PathWithContents> {
        let mut paths = Vec::new();
        if let Some(ref template) = self.system_template {
            paths.push(template);
        }
        if let Some(ref template) = self.user_template {
            paths.push(template);
        }
        if let Some(ref template) = self.assistant_template {
            paths.push(template);
        }
        paths
    }
}

