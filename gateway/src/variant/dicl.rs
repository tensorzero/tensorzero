use serde::de::{Deserializer, Error as SerdeError};
use std::fs;
use std::{collections::HashMap, path::PathBuf};

use serde::Deserialize;

use crate::embeddings::EmbeddingResponseWithMetadata;
use crate::endpoints::inference::InferenceModels;
use crate::inference::types::{ModelInferenceRequest, RequestMessage, Role};
use crate::{
    embeddings::{EmbeddingModelConfig, EmbeddingRequest},
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

use super::{
    infer_model_request, infer_model_request_stream, prepare_model_inference_request,
    InferenceConfig, JsonMode, ModelUsedInfo, Variant,
};

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
        models: &'request InferenceModels<'a>,
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

        // TODO (Viraj): get the embedding response into observability
        let (relevant_examples, embedding_response) = self
            .retrieve_relevant_examples(
                serialized_input,
                &models.embedding_models,
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
        let model_config = models.models.get(&self.model).ok_or(Error::UnknownModel {
            name: self.model.clone(),
        })?;
        let mut inference_response = infer_model_request(
            model_inference_request,
            &self.model,
            model_config,
            function,
            inference_config,
            clients,
            inference_params,
        )
        .await?;
        inference_response
            .mut_model_inference_results()
            .push(embedding_response.into());
        Ok(inference_response)
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
        let mut inference_params = inference_params;
        let serialized_input = serde_json::to_string(&input).map_err(|e| Error::Serialization {
            message: format!(
                "Error in serializing Input in dynamic in-context learning variant: {}",
                e
            ),
        })?;

        // TODO (Viraj): get the embedding response into observability
        let (relevant_examples, embedding_response) = self
            .retrieve_relevant_examples(
                serialized_input,
                &models.embedding_models,
                clients,
                &inference_config.function_name,
                &inference_config.variant_name,
            )
            .await?;
        let request = self.prepare_request(
            &input,
            &relevant_examples,
            function,
            inference_config,
            true,
            &mut inference_params,
        )?;
        let model_config = models.models.get(&self.model).ok_or(Error::UnknownModel {
            name: self.model.clone(),
        })?;
        let (inference_result_chunk, inference_result_stream, mut model_used_info) =
            infer_model_request_stream(
                request,
                &self.model,
                model_config,
                function,
                clients,
                inference_params,
            )
            .await?;
        model_used_info
            .previous_model_inference_results
            .push(embedding_response.into());
        Ok((
            inference_result_chunk,
            inference_result_stream,
            model_used_info,
        ))
    }

    fn validate(
        &self,
        _function: &FunctionConfig,
        models: &HashMap<String, ModelConfig>,
        embedding_models: &HashMap<String, EmbeddingModelConfig>,
        _templates: &TemplateConfig,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(), Error> {
        // TODO (#360): Add the clickhouse connection to this interface
        // Run a count() query on the DynamicInContextLearningExample table
        // WHERE function_name = function_name and variant_name = variant_name
        // Make sure that the count is positive

        // Validate that weight is non-negative
        if self.weight < 0.0 {
            return Err(Error::Config {
                message: format!(
                "`functions.{function_name}.variants.{variant_name}`: `weight` must be non-negative"
            ),
            });
        }
        // Validate that the generation model and embedding model are valid
        let model = models.get(&self.model).ok_or_else(|| Error::Config {
            message: format!("`functions.{function_name}.variants.{variant_name}`: `model` must be a valid model name"),
        })?;
        let embedding_model = embedding_models.get(&self.embedding_model).ok_or_else(|| Error::Config {
            message: format!("`functions.{function_name}.variants.{variant_name}`: `embedding_model` must be a valid embedding model name"),
        })?;
        model.validate().map_err(|e| Error::Config {
            message: format!(
                "`functions.{function_name}.variants.{variant_name}` and model `{}`: {e}",
                self.model
            ),
        })?;
        embedding_model.validate().map_err(|e| Error::Config {
            message: format!(
                "`functions.{function_name}.variants.{variant_name}` and embedding model `{}`: {e}",
                self.embedding_model
            ),
        })?;
        Ok(())
    }

    fn get_all_template_paths(&self) -> Vec<&PathBuf> {
        vec![]
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
    async fn retrieve_relevant_examples<'a>(
        &'a self,
        serialized_input: String,
        embedding_models: &'a HashMap<String, EmbeddingModelConfig>,
        clients: &InferenceClients<'_>,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(Vec<Example>, EmbeddingResponseWithMetadata<'a>), Error> {
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
        let embedding_response_with_metadata =
            EmbeddingResponseWithMetadata::new(embedding_reponse, &self.embedding_model);
        let formatted_embedding = format!(
            "[{}]",
            embedding_response_with_metadata
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
        Ok((examples, embedding_response_with_metadata))
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

    fn prepare_request<'a, 'request>(
        &'a self,
        input: &Input,
        examples: &[Example],
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        stream: bool,
        inference_params: &mut InferenceParams,
    ) -> Result<ModelInferenceRequest<'request>, Error>
    where
        'a: 'request,
    {
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
