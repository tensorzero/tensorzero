use std::path::Path;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::config_parser::PathWithContents;
use crate::embeddings::EmbeddingModelTable;
use crate::endpoints::inference::{InferenceClients, InferenceModels, InferenceParams};
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use crate::function::FunctionConfig;
use crate::inference::types::batch::StartBatchModelInferenceWithMetadata;
use crate::inference::types::{
    ContentBlockOutput, InferenceResult, InferenceResultStream, InternalJsonInferenceOutput,
    JsonInferenceResult, ResolvedInput, Thought,
};
use crate::jsonschema_util::DynamicJSONSchema;
use crate::minijinja_util::TemplateConfig;
use crate::model::ModelTable;
use crate::{
    config_parser::LoadableConfig,
    variant::chat_completion::{ChatCompletionConfig, UninitializedChatCompletionConfig},
};

use super::{InferenceConfig, ModelUsedInfo, Variant};

#[derive(Debug)]
pub struct ChainOfThoughtConfig {
    pub inner: ChatCompletionConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedChainOfThoughtConfig {
    #[serde(flatten)]
    pub inner: UninitializedChatCompletionConfig,
}

impl LoadableConfig<ChainOfThoughtConfig> for UninitializedChainOfThoughtConfig {
    fn load<P: AsRef<Path>>(self, base_path: P) -> Result<ChainOfThoughtConfig, Error> {
        Ok(ChainOfThoughtConfig {
            inner: self.inner.load(base_path)?,
        })
    }
}

impl Variant for ChainOfThoughtConfig {
    async fn infer<'a: 'request, 'request>(
        &self,
        input: &ResolvedInput,
        models: &'request InferenceModels<'a>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'static, 'request>,
        clients: &'request InferenceClients<'request>,
        inference_params: InferenceParams,
    ) -> Result<InferenceResult, Error> {
        let FunctionConfig::Json(json_config) = function else {
            // This should never happen, because we check this in `validate`
            return Err(ErrorDetails::Inference {
                message: format!(
                    "Can't use chain of thought variant type with non-JSON function. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            }
            .into());
        };
        let original_output_schema = match inference_config.dynamic_output_schema {
            Some(schema) => &schema.value,
            None => json_config.output_schema.value,
        };
        let augmented_output_schema = prepare_thinking_output_schema(original_output_schema);
        let augmented_inference_config = InferenceConfig {
            dynamic_output_schema: Some(&augmented_output_schema),
            tool_config: None, // Dynamic tool configs are handled farther down, we don't need to set that here
            templates: inference_config.templates,
            function_name: inference_config.function_name,
            variant_name: inference_config.variant_name,
            ids: inference_config.ids,
            extra_body: inference_config.extra_body.clone(),
            extra_cache_key: inference_config.extra_cache_key.clone(),
            extra_headers: inference_config.extra_headers.clone(),
        };
        let inference_result = self
            .inner
            .infer(
                input,
                models,
                function,
                &augmented_inference_config,
                clients,
                inference_params,
            )
            .await?;
        let InferenceResult::Json(json_result) = inference_result else {
            return Err(ErrorDetails::Inference {
                message: format!(
                    "Chain of thought variant can only be used with JSON functions. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            }
            .into());
        };
        let output = parse_thinking_output(json_result.output)?;
        Ok(InferenceResult::Json(JsonInferenceResult {
            inference_id: json_result.inference_id,
            created: json_result.created,
            output,
            usage: json_result.usage,
            model_inference_results: json_result.model_inference_results,
            output_schema: original_output_schema.clone(),
            inference_params: json_result.inference_params,
            original_response: json_result.original_response,
            finish_reason: json_result.finish_reason,
        }))
    }

    async fn infer_stream<'request>(
        &self,
        _input: &ResolvedInput,
        _models: &'request InferenceModels<'_>,
        _function: &FunctionConfig,
        _inference_config: &'request InferenceConfig<'static, 'request>,
        _clients: &'request InferenceClients<'request>,
        _inference_params: InferenceParams,
    ) -> Result<(InferenceResultStream, ModelUsedInfo), Error> {
        Err(ErrorDetails::UnsupportedVariantForStreamingInference {
            variant_type: "chain_of_thought".to_string(),
            issue_link: Some("https://github.com/tensorzero/tensorzero/issues/1839".to_string()),
        }
        .into())
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
        if !matches!(function, FunctionConfig::Json(_)) {
            return Err(ErrorDetails::UnsupportedVariantForFunctionType {
                function_name: function_name.to_string(),
                variant_name: variant_name.to_string(),
                function_type: "chat".to_string(),
                variant_type: "chain_of_thought".to_string(),
            }
            .into());
        }
        self.inner
            .validate(
                function,
                models,
                embedding_models,
                templates,
                function_name,
                variant_name,
            )
            .await
    }

    fn get_all_template_paths(&self) -> Vec<&PathWithContents> {
        self.inner.get_all_template_paths()
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _input: &[ResolvedInput],
        _models: &'a InferenceModels<'a>,
        _function: &'a FunctionConfig,
        _inference_configs: &'a [InferenceConfig<'a, 'a>],
        _clients: &'a InferenceClients<'a>,
        _inference_params: Vec<InferenceParams>,
    ) -> Result<StartBatchModelInferenceWithMetadata<'a>, Error> {
        Err(ErrorDetails::UnsupportedVariantForBatchInference { variant_name: None }.into())
    }
}

/// Converts the output schema of the actual function being called into a schema that enforces chain
/// of thought reasoning.
fn prepare_thinking_output_schema(previous_output_schema: &Value) -> DynamicJSONSchema {
    DynamicJSONSchema::new(json!({
        "type": "object",
        "properties": {
            "thinking": {
                "type": "string",
                "description": "A detailed description of the thought process used to arrive at the final answer.",
            },
            "response": previous_output_schema,
        },
        "required": ["thinking", "response"],
        "additionalProperties": false,
    }))
}

/// Parses the output of the actual function being called and serializes the `response` field.
/// After this point, the function output should look like a normal JSON response and not include thinking.
/// It also adds the thinking to the auxiliary content at the index the original JSON block was at.
fn parse_thinking_output(
    mut output: InternalJsonInferenceOutput,
) -> Result<InternalJsonInferenceOutput, Error> {
    match &mut output.parsed {
        // If the output failed to parse, don't handle it here.
        None => Ok(output),
        Some(parsed) => {
            let Some(thinking) = parsed
                .get_mut("thinking")
                .and_then(|v| v.as_str().map(|s| s.to_string()))
            else {
                tracing::warn!(
                    "Chain of thought variant received a parsed output that didn't contain a `thinking` field. {}",
                    IMPOSSIBLE_ERROR_MESSAGE
                );
                return Ok(output);
            };
            let Some(response) = parsed.get_mut("response") else {
                tracing::warn!(
                    "Chain of thought variant received a parsed output that didn't contain a `response` field. {}",
                    IMPOSSIBLE_ERROR_MESSAGE
                );
                return Ok(output);
            };
            let serialized_response =
                serde_json::to_string(response).map_err(|e| ErrorDetails::Serialization {
                    message: format!("Failed to serialize chain of thought response: {e}"),
                })?;
            output.parsed = Some(response.take());
            output.raw = Some(serialized_response);
            // Insert the thinking at the index the original JSON block was at
            let Some(json_block_index) = output.json_block_index else {
                tracing::warn!(
                    "Chain of thought variant received a parsed output that didn't contain a `json_block_index` field. {}",
                    IMPOSSIBLE_ERROR_MESSAGE
                );
                output
                    .auxiliary_content
                    .push(ContentBlockOutput::Thought(Thought {
                        text: thinking,
                        signature: None,
                    }));
                return Ok(output);
            };
            output.auxiliary_content.insert(
                json_block_index,
                ContentBlockOutput::Thought(Thought {
                    text: thinking,
                    signature: None,
                }),
            );
            Ok(output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prepare_thinking_output_schema() {
        let previous_output_schema = json!({
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                },
            },
        });
        let thinking_output_schema = prepare_thinking_output_schema(&previous_output_schema);
        assert_eq!(
            thinking_output_schema.value,
            json!({
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": "A detailed description of the thought process used to arrive at the final answer.",
                    },
                    "response": previous_output_schema,
                },
                "required": ["thinking", "response"],
                "additionalProperties": false,
            })
        );
    }

    #[test]
    fn test_parse_thinking_output() {
        use serde_json::json;

        // Case 1: parsed is None (should return output unchanged)
        let output = InternalJsonInferenceOutput {
            raw: Some("raw string".to_string()),
            parsed: None,
            auxiliary_content: vec![],
            json_block_index: Some(0),
        };
        let result = parse_thinking_output(output.clone());
        assert!(result.is_ok());
        assert_eq!(result.unwrap().raw, Some("raw string".to_string()));

        // Case 2: parsed is Some, but no 'response' field (should warn and return output unchanged)
        let output = InternalJsonInferenceOutput {
            raw: Some("raw string".to_string()),
            parsed: Some(json!({"not_response": 123})),
            auxiliary_content: vec![],
            json_block_index: Some(0),
        };
        let result = parse_thinking_output(output.clone());
        assert!(result.is_ok());
        let out = result.unwrap();
        // Should still have the original parsed value
        assert_eq!(out.parsed, Some(json!({"not_response": 123})));
        assert_eq!(out.raw, Some("raw string".to_string()));

        // Case 3: parsed is Some, 'response' field is present and serializable
        let output = InternalJsonInferenceOutput {
            raw: None,
            parsed: Some(json!({
                "thinking": "step by step",
                "response": {"answer": "42"}
            })),
            auxiliary_content: vec![],
            json_block_index: Some(0),
        };
        let result = parse_thinking_output(output);
        assert!(result.is_ok());
        let out = result.unwrap();
        // The parsed should now be just the response object
        assert_eq!(out.parsed, Some(json!({"answer": "42"})));
        // The raw should be the serialized response
        assert_eq!(out.raw, Some("{\"answer\":\"42\"}".to_string()));
        // The auxiliary content should now contain the thinking
        assert_eq!(out.auxiliary_content.len(), 1);
        assert_eq!(
            out.auxiliary_content[0],
            ContentBlockOutput::Thought(Thought {
                text: "step by step".to_string(),
                signature: None,
            })
        );

        // Case 4: There is already existing thinking in auxiliary_content and a well-formed output with thinking field
        let output = InternalJsonInferenceOutput {
            raw: None,
            parsed: Some(json!({
                "thinking": "new thinking process",
                "response": {"answer": "the ultimate answer is 42"}
            })),
            auxiliary_content: vec![ContentBlockOutput::Thought(Thought {
                text: "existing thinking".to_string(),
                signature: None,
            })],
            json_block_index: Some(0),
        };
        let result = parse_thinking_output(output);
        assert!(result.is_ok());
        let out = result.unwrap();
        // The parsed should now be just the response object
        assert_eq!(
            out.parsed,
            Some(json!({"answer": "the ultimate answer is 42"}))
        );
        // The raw should be the serialized response
        assert_eq!(
            out.raw,
            Some("{\"answer\":\"the ultimate answer is 42\"}".to_string())
        );
        // The auxiliary content should now contain both the existing and new thinking
        assert_eq!(out.auxiliary_content.len(), 2);
        // The new thinking should come first since the json block index is 0
        // (it came from the first content block that was generated so we should put it back there)
        assert_eq!(
            out.auxiliary_content[0],
            ContentBlockOutput::Thought(Thought {
                text: "new thinking process".to_string(),
                signature: None,
            })
        );
        assert_eq!(
            out.auxiliary_content[1],
            ContentBlockOutput::Thought(Thought {
                text: "existing thinking".to_string(),
                signature: None,
            })
        );
    }
}
