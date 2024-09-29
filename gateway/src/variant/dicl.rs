use std::fs;
use std::path::Path;
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

#[derive(Debug, Deserialize)]
pub struct UninitializedDiclConfig {
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

        let (relevant_examples, embedding_response) = self
            .retrieve_relevant_examples(
                serialized_input,
                models.embedding_models,
                clients,
                &inference_config.function_name,
                &inference_config.variant_name,
            )
            .await?;
        let model_inference_request = self.prepare_request(
            input,
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

        let (relevant_examples, embedding_response) = self
            .retrieve_relevant_examples(
                serialized_input,
                models.embedding_models,
                clients,
                &inference_config.function_name,
                &inference_config.variant_name,
            )
            .await?;
        let request = self.prepare_request(
            input,
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

#[derive(Debug, Deserialize, PartialEq)]
struct ChatExample {
    input: Input,
    output: Vec<ContentBlockOutput>,
    // distance: Option<f32>, // Include if needed
}

// Start of Selection
#[derive(Debug, Deserialize, PartialEq)]
struct JsonExample {
    input: Input,
    output: JsonInferenceOutput,
    // distance: Option<f32>, // Include if needed
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
    // Include `distance` if you need it later
    // distance: f32,
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
            .embed(&embedding_request, clients.http_client)
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

        // Parse each line into RawExample
        let raw_examples: Vec<RawExample> = result
            .lines()
            .map(serde_json::from_str::<RawExample>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| Error::Serialization {
                message: format!("Failed to parse raw examples: {}", e),
            })?;

        // Convert RawExamples into Examples
        let examples = parse_raw_examples(raw_examples)?;

        if examples.len() != self.k as usize {
            tracing::warn!(
                "Dynamic in-context learning retrieved {} examples, expected {}",
                examples.len(),
                self.k
            );
        }

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

// Since the `input` and `output` fields in the ClickHouse table are of type String,
// we need to parse them into the appropriate types before using them and cannot rely
// on Deserialize to do it for us.
fn parse_raw_examples(raw_examples: Vec<RawExample>) -> Result<Vec<Example>, Error> {
    let mut examples = Vec::new();
    for raw_example in raw_examples {
        // Parse the `input` string into `Input`
        let input: Input =
            serde_json::from_str(&raw_example.input).map_err(|e| Error::Serialization {
                message: format!("Failed to parse `input`: {}", e),
            })?;

        // Try parsing `output` as `Vec<ContentBlockOutput>` (for ChatExample)
        if let Ok(output) = serde_json::from_str::<Vec<ContentBlockOutput>>(&raw_example.output) {
            examples.push(Example::Chat(ChatExample { input, output }));
            continue;
        }

        // Try parsing `output` as `JsonInferenceOutput` (for JsonExample)
        if let Ok(output) = serde_json::from_str::<JsonInferenceOutput>(&raw_example.output) {
            examples.push(Example::Json(JsonExample { input, output }));
            continue;
        }

        // If neither parsing succeeded, return an error
        return Err(Error::Serialization {
            message: "Failed to parse `output` into known formats".to_string(),
        });
    }

    Ok(examples)
}

impl UninitializedDiclConfig {
    pub fn load<P: AsRef<Path>>(self, base_path: P) -> Result<DiclConfig, Error> {
        let system_instructions = match self.system_instructions {
            Some(path) => {
                let path = base_path.as_ref().join(path);
                fs::read_to_string(path).map_err(|e| Error::Config {
                    message: format!("Failed to read system instructions: {}", e),
                })?
            }
            None => "You are tasked with learning by induction and then solving a problem below. You will be shown several examples of inputs followed by outputs. Then, in the same format you will be given one last set of inputs. Your job is to use the provided examples to inform your response to the last set of inputs. ".to_string(),
        };

        Ok(DiclConfig {
            weight: self.weight,
            embedding_model: self.embedding_model,
            k: self.k,
            model: self.model,
            system_instructions,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            seed: self.seed,
            json_mode: self.json_mode,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::*;
    use crate::tool::*;
    use serde_json::json;

    #[test]
    fn test_prepare_message() {
        // Create an instance of DiclConfig (assuming default implementation is available)
        let dicl_config = DiclConfig::default();

        // ---------- Test with ChatExample ----------

        // Mock Input data
        let input_data = Input {
            system: Some(json!({"type": "system", "content": "System message"})),
            messages: vec![
                InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text {
                        value: json!("Hello, assistant!"),
                    }],
                },
                InputMessage {
                    role: Role::Assistant,
                    content: vec![InputMessageContent::Text {
                        value: json!("Hello, user!"),
                    }],
                },
            ],
        };

        // Mock Output data for ChatExample
        let chat_output = vec![
            ContentBlockOutput::Text(Text {
                text: "This is a test response.".to_string(),
            }),
            ContentBlockOutput::ToolCall(ToolCallOutput {
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

        let chat_messages = dicl_config.prepare_message(&chat_example).unwrap();

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
        let expected_content: Vec<ContentBlock> =
            chat_output.into_iter().map(|x| x.into()).collect();

        assert_eq!(chat_messages[1].role, Role::Assistant);
        assert_eq!(chat_messages[1].content, expected_content);

        // ---------- Test with JsonExample ----------

        // Mock Output data for JsonExample
        let json_output = JsonInferenceOutput {
            raw: "{\"result\": \"success\"}".to_string(),
            parsed: Some(json!({"result": "success"})),
        };

        let json_example = Example::Json(JsonExample {
            input: input_data.clone(),
            output: json_output.clone(),
        });

        let json_messages = dicl_config.prepare_message(&json_example).unwrap();

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
            text: json_output.raw.clone(),
        })];

        assert_eq!(json_messages[1].role, Role::Assistant);
        assert_eq!(json_messages[1].content, expected_content);
    }

    #[test]
    fn test_prepare_input_message() {
        // Create an instance of DiclConfig (assuming default implementation is available)
        let dicl_config = DiclConfig::default();

        // Mock Input data
        let input_data = Input {
            system: Some(json!({"type": "system", "content": "System message"})),
            messages: vec![
                InputMessage {
                    role: Role::User,
                    content: vec![
                        InputMessageContent::Text {
                            value: json!("Hello, assistant!"),
                        },
                        InputMessageContent::ToolCall(ToolCall {
                            id: "tool_call_1".to_string(),
                            name: "search_tool".to_string(),
                            arguments: "{\"query\": \"rust programming\"}".to_string(),
                        }),
                    ],
                },
                InputMessage {
                    role: Role::Assistant,
                    content: vec![InputMessageContent::Text {
                        value: json!("Here are the search results for rust programming."),
                    }],
                },
            ],
        };

        // Call the prepare_input_message function
        let request_message = dicl_config.prepare_input_message(&input_data).unwrap();

        // The role should be User
        assert_eq!(request_message.role, Role::User);

        // The content should contain the serialized Input as a Text ContentBlock
        let expected_serialized_input = serde_json::to_string(&input_data).unwrap();
        let expected_content = vec![ContentBlock::Text(Text {
            text: expected_serialized_input.clone(),
        })];
        assert_eq!(request_message.content, expected_content);
    }

    #[test]
    fn test_parse_raw_examples() {
        // Define sample raw examples with serialized Input and Output
        let raw_examples = vec![
            RawExample {
                input: serde_json::to_string(&Input {
                    system: Some(json!({"assistant_name": "Dr. Mehta"})),
                    messages: vec![InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text {
                            value: json!("What is the boiling point of water?"),
                        }],
                    }],
                })
                .unwrap(),
                output: serde_json::to_string(&vec![ContentBlockOutput::Text(Text {
                    text: "100 degrees Celsius".to_string(),
                })])
                .unwrap(),
            },
            RawExample {
                input: serde_json::to_string(&Input {
                    system: Some(json!({"assistant_name": "Pinocchio"})),
                    messages: vec![InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text {
                            value: json!("What is the capital city of Japan?"),
                        }],
                    }],
                })
                .unwrap(),
                output: serde_json::to_string(&vec![ContentBlockOutput::Text(Text {
                    text: "Osaka (nose grows 4 inches)".to_string(),
                })])
                .unwrap(),
            },
        ];

        // Parse the raw examples
        let parsed_examples =
            parse_raw_examples(raw_examples.clone()).expect("Failed to parse raw examples");

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
                input: serde_json::to_string(&Input {
                    system: Some(json!({"assistant_name": "JsonTester"})),
                    messages: vec![InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text {
                            value: json!("Provide a sample JSON response."),
                        }],
                    }],
                })
                .unwrap(),
                output: serde_json::to_string(&JsonInferenceOutput {
                    raw: "{\"status\": \"success\", \"data\": {\"id\": 1}}".to_string(),
                    parsed: Some(json!({
                        "status": "success",
                        "data": {
                            "id": 1
                        }
                    })),
                })
                .unwrap(),
            },
            RawExample {
                input: serde_json::to_string(&Input {
                    system: Some(json!({"assistant_name": "JsonTester"})),
                    messages: vec![InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text {
                            value: json!("Provide another JSON response."),
                        }],
                    }],
                })
                .unwrap(),
                output: serde_json::to_string(&JsonInferenceOutput {
                    raw: "{\"result\": [1, 2, 3], \"status\": \"ok\"}".to_string(),
                    parsed: Some(json!({
                        "result": [1, 2, 3],
                        "status": "ok"
                    })),
                })
                .unwrap(),
            },
        ];

        // Parse the JSON raw examples
        let parsed_json_examples = parse_raw_examples(json_raw_examples.clone())
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
}
