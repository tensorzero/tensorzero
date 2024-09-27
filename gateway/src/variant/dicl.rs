use serde::de::{Deserializer, Error as SerdeError};
use std::borrow::Cow;
use std::fs;
use std::iter::once;
use std::{collections::HashMap, path::PathBuf};

use reqwest::Client;
use serde::Deserialize;

use crate::inference::types::{ModelInferenceRequest, RequestMessage, Role};
use crate::{
    clickhouse::ClickHouseConnectionInfo,
    embeddings::{EmbeddingModelConfig, EmbeddingRequest, EmbeddingResponse},
    endpoints::inference::{InferenceClients, InferenceParams},
    error::Error,
    function::FunctionConfig,
    inference::types::{
        ContentBlock, ContentBlockOutput, InferenceResult, InferenceResultChunk,
        InferenceResultStream, Input, JsonInferenceOutput,
    },
    minijinja_util::TemplateConfig,
    model::ModelConfig,
};

use super::{prepare_model_inference_request, InferenceConfig, JsonMode, ModelUsedInfo, Variant};

#[derive(Debug, Default)]
pub struct DiclConfig {
    pub weight: f64,
    pub embedding_model: String,
    pub k: u32, // k as in k-nearest neighbors
    pub model: String,
    pub system_instructions: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub seed: Option<u32>,
    pub json_mode: JsonMode,
}

impl Variant for DiclConfig {
    async fn infer<'a, 'request>(
        &'a self,
        input: &Input,
        models: &'a HashMap<String, ModelConfig>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        clients: &'request InferenceClients<'request>,
        inference_params: InferenceParams,
    ) -> Result<InferenceResult, Error> {
        let mut inference_params = inference_params;
        let serialized_input = serde_json::to_string(&input).map_err(|e| Error::Serialization {
            message: format!(
                "Error in serializing Input in dynamic in-context learning variant: {}",
                e
            ),
        })?;

        let (relevant_examples, embedding_response) = self
            .retrieve_relevant_examples(
                serialized_input,
                &inference_config.embedding_models,
                clients,
                &inference_config.function_name,
                &inference_config.variant_name,
            )
            .await?;
        let model_inference_request = self.prepare_request(
            &input,
            &relevant_examples,
            function,
            inference_config,
            false,
            &mut inference_params,
        )?;

        todo!()
    }

    async fn infer_stream<'request>(
        &'static self,
        input: &Input,
        models: &'static HashMap<String, ModelConfig>,
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
        todo!()
    }

    fn validate(
        &self,
        function: &FunctionConfig,
        models: &HashMap<String, ModelConfig>,
        templates: &TemplateConfig,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(), Error> {
        todo!()
    }

    fn get_all_template_paths(&self) -> Vec<&PathBuf> {
        todo!()
    }
}

#[derive(Debug, Deserialize)]
struct ChatExample {
    input: Input,
    output: Vec<ContentBlockOutput>,
    distance: f32,
}

#[derive(Debug, Deserialize)]
struct JsonExample {
    input: Input,
    output: JsonInferenceOutput,
    distance: f32,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Example {
    Chat(ChatExample),
    Json(JsonExample),
}

impl DiclConfig {
    async fn retrieve_relevant_examples(
        &self,
        serialized_input: String,
        embedding_models: &HashMap<String, EmbeddingModelConfig>,
        clients: &InferenceClients<'_>,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(Vec<Example>, EmbeddingResponse), Error> {
        let embedding_model =
            embedding_models
                .get(&self.embedding_model)
                .ok_or(Error::Inference {
                    message: format!("Embedding model {} not found", self.embedding_model),
                })?;
        let embedding_request = EmbeddingRequest {
            input: serialized_input.to_string(),
        };
        let embedding_reponse = embedding_model
            .embed(&embedding_request, &clients.http_client)
            .await?;
        let formatted_embedding = format!(
            "[{}]",
            embedding_reponse
                .embedding
                .iter()
                .map(|&x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let query = format!(
            r#"SELECT input, output, cosineDistance(embedding, {}) as distance
               FROM DynamicInContextLearningExample
               WHERE function_name='{}' AND variant_name='{}'
               ORDER BY distance ASC
               LIMIT {}
               FORMAT JSONEachRow"#,
            formatted_embedding, function_name, variant_name, self.k
        );
        let result = clients.clickhouse_connection_info.run_query(query).await?;
        let examples: Vec<Example> = result
            .lines()
            .map(|line| serde_json::from_str::<Example>(line))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| Error::Serialization {
                message: format!("Failed to parse examples: {}", e),
            })?;
        Ok((examples, embedding_reponse))
    }

    /// Serialize an example into a pair of RequestMessages
    /// The first message is a User message with the input serialized
    /// The second message is an Assistant message with the output as native output blocks
    ///   - For chat messages, this is a simple vector of ContentBlocks
    ///   - For JSON messages, this is a single JSON output block (as Text)
    fn prepare_message(&self, example: &Example) -> Result<Vec<RequestMessage>, Error> {
        let mut messages = Vec::new();
        let input = match example {
            Example::Chat(chat_example) => chat_example.input.clone(),
            Example::Json(json_example) => json_example.input.clone(),
        };
        messages.push(RequestMessage {
            role: Role::User,
            content: vec![serde_json::to_string(&input)
                .map_err(|e| Error::Serialization {
                    message: format!(
                        "Error in serializing Input in dynamic in-context learning variant: {}",
                        e
                    ),
                })?
                .into()],
        });
        let content: Vec<ContentBlock> = match example {
            Example::Chat(chat_example) => chat_example
                .output
                .clone()
                .into_iter()
                .map(|x| x.into())
                .collect(),
            Example::Json(json_example) => vec![json_example.output.raw.clone().into()],
        };
        messages.push(RequestMessage {
            role: Role::Assistant,
            content,
        });
        Ok(messages)
    }

    fn prepare_input_message(&self, input: &Input) -> Result<RequestMessage, Error> {
        let content = vec![serde_json::to_string(&input)
            .map_err(|e| Error::Serialization {
                message: format!(
                    "Error in serializing Input in dynamic in-context learning variant: {}",
                    e
                ),
            })?
            .into()];
        Ok(RequestMessage {
            role: Role::User,
            content,
        })
    }

    fn prepare_request<'a>(
        &self,
        input: &Input,
        examples: &[Example],
        function: &'a FunctionConfig,
        inference_config: &'a InferenceConfig<'a>,
        stream: bool,
        inference_params: &mut InferenceParams,
    ) -> Result<ModelInferenceRequest<'a>, Error> {
        let messages = examples
            .iter()
            .map(|example| self.prepare_message(example))
            .collect::<Result<Vec<Vec<RequestMessage>>, _>>()?
            .into_iter()
            .flatten()
            .chain(std::iter::once(self.prepare_input_message(input)?))
            .collect::<Vec<_>>();
        let system = Some(self.system_instructions.clone());
        inference_params
            .chat_completion
            .backfill_with_variant_params(self.temperature, self.max_tokens, self.seed);
        prepare_model_inference_request(
            messages,
            system,
            function,
            inference_config,
            stream,
            inference_params,
            &self.json_mode,
        )
    }
}

impl<'de> Deserialize<'de> for DiclConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct DiclConfigHelper {
            #[serde(default)]
            pub weight: f64,
            pub embedding_model: String,
            pub k: u32, // k as in k-nearest neighbors
            pub model: String,
            pub system_instructions: Option<PathBuf>,
            pub temperature: Option<f32>,
            pub max_tokens: Option<u32>,
            pub seed: Option<u32>,
            #[serde(default)]
            pub json_mode: JsonMode,
        }

        let helper = DiclConfigHelper::deserialize(deserializer)?;

        let system_instructions = match helper.system_instructions {
            Some(path) => {
                let contents = fs::read_to_string(&path).map_err(SerdeError::custom)?;
                // TODO: Replace the placeholder with actual template formatting
                format!("Template formatted with contents: {}", contents)
            }
            None => "Default system instructions".to_string(),
        };

        Ok(DiclConfig {
            weight: helper.weight,
            embedding_model: helper.embedding_model,
            k: helper.k,
            model: helper.model,
            system_instructions,
            temperature: helper.temperature,
            max_tokens: helper.max_tokens,
            seed: helper.seed,
            json_mode: helper.json_mode,
        })
    }
}
