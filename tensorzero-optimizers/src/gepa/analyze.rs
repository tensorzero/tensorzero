//! Inference analysis functions for GEPA optimization
//!
//! This module provides functions for:
//! - Analyzing inference outputs using the analyze function
//! - Parsing analysis responses
//! - Building analysis inputs

use std::collections::HashMap;

use serde::Serialize;

use tensorzero_core::{
    client::{
        Client, ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
        InferenceOutput,
    },
    config::{path::ResolvedTomlPath, UninitializedVariantConfig, UninitializedVariantInfo},
    endpoints::{datasets::StoredDatapoint, inference::InferenceResponse},
    error::{Error, ErrorDetails},
    function::FunctionConfig,
    inference::types::{ContentBlockChatOutput, Role, TextKind},
    variant::chat_completion::UninitializedChatCompletionConfig,
};

use evaluations::stats::EvaluationInfo;

/// Helper function to serialize values with consistent error handling
fn serialize_to_value<T: serde::Serialize>(
    value: &T,
    context: &str,
) -> Result<serde_json::Value, Error> {
    serde_json::to_value(value).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize {context}: {e}"),
        })
    })
}

/// Helper function to serialize values to strings with consistent error handling
fn serialize_to_string<T: serde::Serialize>(value: &T, context: &str) -> Result<String, Error> {
    serde_json::to_string(value).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize {context}: {e}"),
        })
    })
}

// TODO: The mutate.rs module has similar patterns that could use these helpers.
// Consider extracting to a shared gepa/helpers.rs module if duplication becomes problematic.

/// Represents an inference output paired with its analysis feedback
#[derive(Debug, Clone, Serialize)]
pub struct InferenceWithAnalysis {
    pub inference_output: InferenceResponse,
    /// Raw XML text output from the analyze function
    pub analysis: String,
}

/// Build the input JSON for the analyze function
pub fn build_analyze_input(
    eval_info: &EvaluationInfo,
    function_config: &FunctionConfig,
    variant_config: &UninitializedChatCompletionConfig,
) -> Result<serde_json::Value, Error> {
    // Extract all templates from templates.inner HashMap using idiomatic iterators
    let templates_map: HashMap<String, String> = variant_config
        .templates
        .inner
        .iter()
        .map(|(name, config)| config.path.read().map(|content| (name.clone(), content)))
        .collect::<Result<_, _>>()?;

    // Extract schemas from function config
    let system_schema = function_config.system_schema().map(|s| s.value.clone());
    let user_schema = function_config.user_schema().map(|s| s.value.clone());
    let assistant_schema = function_config.assistant_schema().map(|s| s.value.clone());

    // Extract output schema for JSON functions
    let output_schema = match function_config {
        FunctionConfig::Json(params) => Some(params.output_schema.value.clone()),
        FunctionConfig::Chat(_) => None,
    };

    // Extract tools for Chat functions
    let tools = match function_config {
        FunctionConfig::Chat(params) => {
            if params.tools.is_empty() {
                None
            } else {
                Some(serde_json::json!(params.tools))
            }
        }
        FunctionConfig::Json(_) => None,
    };

    // Serialize the inference output
    let output = serialize_to_value(&eval_info.response, "inference response")?;

    // Get tags from datapoint (if any)
    let tags = match &eval_info.datapoint {
        StoredDatapoint::Chat(dp) => dp.tags.clone(),
        StoredDatapoint::Json(dp) => dp.tags.clone(),
    };

    // Get function name from datapoint
    let function_name = match &eval_info.datapoint {
        StoredDatapoint::Chat(dp) => &dp.function_name,
        StoredDatapoint::Json(dp) => &dp.function_name,
    };

    // Build the input object
    let input = serde_json::json!({
        "function_name": function_name,
        "model": variant_config.model.as_ref(),
        "templates": templates_map,
        "system_schema": system_schema,
        "user_schema": user_schema,
        "assistant_schema": assistant_schema,
        "output_schema": output_schema,
        "tools": tools,
        "tags": tags,
        "input": eval_info.datapoint.input(),
        "output": output,
    });

    Ok(input)
}

/// Parse the analysis response and extract the raw XML text
///
/// Extracts the raw XML text content from the analyze function response.
pub fn parse_analysis_response(response: &InferenceResponse) -> Result<String, Error> {
    let InferenceResponse::Chat(chat_response) = response else {
        return Err(Error::new(ErrorDetails::Inference {
            message: "Expected Chat response from analyze function, got Json".into(),
        }));
    };

    // Extract text content using idiomatic iterator approach
    chat_response
        .content
        .iter()
        .find_map(|block| {
            if let ContentBlockChatOutput::Text(text) = block {
                Some(text.text.clone())
            } else {
                None
            }
        })
        .ok_or_else(|| {
            Error::new(ErrorDetails::Inference {
                message: "No text content found in analyze response".into(),
            })
        })
}

/// Analyze inference outputs using the GEPA analyze function
///
/// Takes evaluation results (datapoint + inference pairs) and calls the built-in
/// `tensorzero::optimization::gepa::analyze` function to get structured feedback.
///
/// Returns a vector of InferenceWithAnalysis containing each inference paired with its analysis.
#[expect(dead_code)]
pub async fn analyze_inferences(
    gateway_client: &Client,
    evaluation_infos: &[EvaluationInfo],
    function_config: &FunctionConfig,
    variant_config: &UninitializedChatCompletionConfig,
    analysis_model: &str,
) -> Result<Vec<InferenceWithAnalysis>, Error> {
    tracing::info!(
        "Analyzing {} inferences using model '{}'",
        evaluation_infos.len(),
        analysis_model
    );

    // Create dynamic variant config for the analyze function
    let analyze_variant_config = UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(UninitializedChatCompletionConfig {
            model: analysis_model.into(),
            weight: None,
            system_template: Some(ResolvedTomlPath::new_fake_path(
                "gepa/analyze/system.minijinja".to_string(),
                include_str!("config/functions/analyze/baseline/system_template.minijinja")
                    .to_string(),
            )),
            user_template: Some(ResolvedTomlPath::new_fake_path(
                "gepa/analyze/user.minijinja".to_string(),
                include_str!("config/functions/analyze/baseline/user_template.minijinja")
                    .to_string(),
            )),
            assistant_template: None,
            ..Default::default()
        }),
        timeouts: None,
    };

    let mut results = Vec::new();

    // Analyze each inference
    for eval_info in evaluation_infos {
        // Build input JSON for the analyze function
        let input_data = build_analyze_input(eval_info, function_config, variant_config)?;

        // Create ClientInferenceParams for the analyze function
        let params = ClientInferenceParams {
            function_name: Some("tensorzero::optimization::gepa::analyze".to_string()),
            model_name: None,
            episode_id: None,
            input: ClientInput {
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: serialize_to_string(&input_data, "analyze input")?,
                    })],
                }],
                system: None,
            },
            stream: None,
            params: Default::default(),
            variant_name: None,
            dryrun: None,
            internal: true,
            tags: HashMap::new(),
            dynamic_tool_params: Default::default(),
            output_schema: None,
            credentials: HashMap::new(),
            cache_options: Default::default(),
            include_original_response: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            internal_dynamic_variant_config: Some(analyze_variant_config.clone()),
            otlp_traces_extra_headers: HashMap::new(),
        };

        // Call the inference API
        let inference_output = gateway_client.inference(params).await.map_err(|e| {
            Error::new(ErrorDetails::Inference {
                message: format!("Failed to call analyze function: {e}"),
            })
        })?;

        // Extract the response
        let response = match inference_output {
            InferenceOutput::NonStreaming(response) => response,
            InferenceOutput::Streaming(_) => {
                return Err(Error::new(ErrorDetails::Inference {
                    message: "Unexpected streaming response from analyze function".to_string(),
                }))
            }
        };

        // Parse the analysis from the response
        let analysis = parse_analysis_response(&response)?;

        results.push(InferenceWithAnalysis {
            inference_output: eval_info.response.clone(),
            analysis,
        });
    }

    Ok(results)
}
