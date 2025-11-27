use chrono::Duration;
use std::collections::HashSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::config::{ErrorContext, PathWithContents, SchemaData};
use crate::embeddings::EmbeddingModelTable;
use crate::endpoints::inference::{InferenceClients, InferenceModels, InferenceParams};
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use crate::function::FunctionConfig;
use crate::inference::types::batch::StartBatchModelInferenceWithMetadata;
use crate::inference::types::resolved_input::LazyResolvedInput;
use crate::inference::types::{
    ContentBlockOutput, InferenceResult, InferenceResultStream, InternalJsonInferenceOutput,
    JsonInferenceResult, Thought,
};
use crate::jsonschema_util::DynamicJSONSchema;
use crate::minijinja_util::TemplateConfig;
use crate::model::ModelTable;
use crate::variant::chat_completion::{ChatCompletionConfig, UninitializedChatCompletionConfig};

use super::{InferenceConfig, ModelUsedInfo, Variant};

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct ChainOfThoughtConfig {
    #[serde(flatten)]
    pub inner: ChatCompletionConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct UninitializedChainOfThoughtConfig {
    #[serde(flatten)]
    pub inner: UninitializedChatCompletionConfig,
}

impl UninitializedChainOfThoughtConfig {
    pub fn load(
        self,
        schemas: &SchemaData,
        error_context: &ErrorContext,
    ) -> Result<ChainOfThoughtConfig, Error> {
        Ok(ChainOfThoughtConfig {
            inner: self.inner.load(schemas, error_context)?,
        })
    }
}

impl ChainOfThoughtConfig {
    /// Converts this initialized config back to its uninitialized form.
    pub fn as_uninitialized(self) -> UninitializedChainOfThoughtConfig {
        UninitializedChainOfThoughtConfig {
            inner: self.inner.as_uninitialized(),
        }
    }
}

impl Variant for ChainOfThoughtConfig {
    async fn infer(
        &self,
        input: Arc<LazyResolvedInput>,
        models: InferenceModels,
        function: Arc<FunctionConfig>,
        inference_config: Arc<InferenceConfig>,
        clients: InferenceClients,
        inference_params: InferenceParams,
    ) -> Result<InferenceResult, Error> {
        let FunctionConfig::Json(json_config) = function.as_ref() else {
            // This should never happen, because we check this in `validate`
            return Err(ErrorDetails::Inference {
                message: format!(
                    "Can't use chain of thought variant type with non-JSON function. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            }
            .into());
        };
        let original_output_schema = match &inference_config.dynamic_output_schema {
            Some(schema) => &schema.value,
            None => &json_config.output_schema.value,
        };
        let augmented_output_schema = prepare_thinking_output_schema(original_output_schema);
        let augmented_inference_config = Arc::new(InferenceConfig {
            dynamic_output_schema: Some(Arc::new(augmented_output_schema)),
            tool_config: None, // Dynamic tool configs are handled farther down, we don't need to set that here
            templates: Arc::clone(&inference_config.templates),
            function_name: Arc::clone(&inference_config.function_name),
            variant_name: Arc::clone(&inference_config.variant_name),
            ids: inference_config.ids,
            fetch_and_encode_input_files_before_inference: inference_config
                .fetch_and_encode_input_files_before_inference,
            extra_body: inference_config.extra_body.clone(),
            extra_cache_key: inference_config.extra_cache_key.clone(),
            extra_headers: inference_config.extra_headers.clone(),
        });
        let inference_result = self
            .inner
            .infer(
                Arc::clone(&input),
                models.clone(),
                Arc::clone(&function),
                augmented_inference_config,
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
            model_inference_results: json_result.model_inference_results,
            output_schema: original_output_schema.clone(),
            inference_params: json_result.inference_params,
            original_response: json_result.original_response,
            finish_reason: json_result.finish_reason,
        }))
    }

    async fn infer_stream(
        &self,
        _input: Arc<LazyResolvedInput>,
        _models: InferenceModels,
        _function: Arc<FunctionConfig>,
        _inference_config: Arc<InferenceConfig>,
        _clients: InferenceClients,
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
        function: Arc<FunctionConfig>,
        models: &ModelTable,
        embedding_models: &EmbeddingModelTable,
        templates: &TemplateConfig<'_>,
        function_name: &str,
        variant_name: &str,
        global_outbound_http_timeout: &Duration,
    ) -> Result<(), Error> {
        if !matches!(function.as_ref(), FunctionConfig::Json(_)) {
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
                Arc::clone(&function),
                models,
                embedding_models,
                templates,
                function_name,
                variant_name,
                global_outbound_http_timeout,
            )
            .await
    }

    fn get_all_template_paths(&self) -> Vec<&PathWithContents> {
        self.inner.get_all_template_paths()
    }

    fn get_all_explicit_template_names(&self) -> HashSet<String> {
        self.inner.get_all_explicit_template_names()
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
                .and_then(|v| v.as_str().map(str::to_string))
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
                        text: Some(thinking),
                        signature: None,
                        summary: None,
                        provider_type: None,
                    }));
                return Ok(output);
            };
            output.auxiliary_content.insert(
                json_block_index,
                ContentBlockOutput::Thought(Thought {
                    text: Some(thinking),
                    signature: None,
                    summary: None,
                    provider_type: None,
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
                text: Some("step by step".to_string()),
                signature: None,
                summary: None,
                provider_type: None,
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
                text: Some("existing thinking".to_string()),
                signature: None,
                summary: None,
                provider_type: None,
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
                text: Some("new thinking process".to_string()),
                signature: None,
                summary: None,
                provider_type: None,
            })
        );
        assert_eq!(
            out.auxiliary_content[1],
            ContentBlockOutput::Thought(Thought {
                text: Some("existing thinking".to_string()),
                signature: None,
                summary: None,
                provider_type: None,
            })
        );
    }

    #[test]
    fn test_as_uninitialized_preserves_basic_fields() {
        let uninitialized = UninitializedChainOfThoughtConfig {
            inner: UninitializedChatCompletionConfig {
                model: "gpt-4".into(),
                weight: Some(0.8),
                temperature: Some(0.7),
                max_tokens: Some(150),
                seed: Some(42),
                ..Default::default()
            },
        };

        let config = uninitialized
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap();

        let exported = config.as_uninitialized();

        assert_eq!(exported.inner.model, "gpt-4".into());
        assert_eq!(exported.inner.weight, Some(0.8));
        assert_eq!(exported.inner.temperature, Some(0.7));
        assert_eq!(exported.inner.max_tokens, Some(150));
        assert_eq!(exported.inner.seed, Some(42));
    }

    #[test]
    fn test_as_uninitialized_preserves_none_values() {
        let uninitialized = UninitializedChainOfThoughtConfig {
            inner: UninitializedChatCompletionConfig {
                model: "gpt-4".into(),
                weight: None,
                temperature: None,
                stop_sequences: None,
                ..Default::default()
            },
        };

        let config = uninitialized
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap();

        let exported = config.as_uninitialized();

        assert_eq!(exported.inner.weight, None);
        assert_eq!(exported.inner.temperature, None);
        assert_eq!(exported.inner.stop_sequences, None);
    }

    #[test]
    fn test_as_uninitialized_serialization_round_trip() {
        let original = UninitializedChainOfThoughtConfig {
            inner: UninitializedChatCompletionConfig {
                model: "gpt-4".into(),
                weight: Some(0.5),
                temperature: Some(0.9),
                ..Default::default()
            },
        };

        let config = original
            .clone()
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap();

        let exported = config.as_uninitialized();

        // Serialize and deserialize
        let json = serde_json::to_string(&exported).unwrap();
        let deserialized: UninitializedChainOfThoughtConfig = serde_json::from_str(&json).unwrap();

        // Should be able to load again
        let reloaded = deserialized
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap();

        assert_eq!(reloaded.inner.model(), &Arc::from("gpt-4"));
        assert_eq!(reloaded.inner.weight(), Some(0.5));
    }
}
