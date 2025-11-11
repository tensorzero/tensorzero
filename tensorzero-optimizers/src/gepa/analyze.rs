//! Inference analysis functions for GEPA optimization
//!
//! This module provides functions for:
//! - Analyzing inference outputs using the analyze function
//! - Parsing analysis responses
//! - Building analysis inputs

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use tensorzero_core::{
    client::{
        Client, ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
        InferenceOutput,
    },
    config::{path::ResolvedTomlPath, UninitializedVariantConfig, UninitializedVariantInfo},
    endpoints::inference::InferenceResponse,
    error::{Error, ErrorDetails},
    function::FunctionConfig,
    inference::types::{ContentBlockChatOutput, Role, TextKind},
    variant::chat_completion::UninitializedChatCompletionConfig,
};

use evaluations::stats::EvaluationInfo;

/// Analysis report from the GEPA analyze function
///
/// The analyze function uses one of three tools to report on inference quality:
/// - Error: Critical failures requiring correction
/// - Improvement: Suboptimal but acceptable outputs that could be better
/// - Optimal: High-quality exemplary outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnalysisReport {
    Error {
        reasoning: String,
        error_identification: String,
        root_cause_analysis: String,
        correct_output: String,
        key_insight: String,
    },
    Improvement {
        reasoning: String,
        suboptimality_description: String,
        better_output: String,
        key_insight: String,
    },
    Optimal {
        reasoning: String,
        key_strengths: Vec<String>,
    },
}

/// Represents an inference output paired with its analysis feedback
#[derive(Debug, Clone, Serialize)]
pub struct InferenceWithAnalysis {
    pub inference_output: InferenceResponse,
    pub analysis: AnalysisReport,
}

/// Build the input JSON for the analyze function
pub fn build_analyze_input(
    eval_info: &EvaluationInfo,
    function_config: &FunctionConfig,
) -> Result<serde_json::Value, Error> {
    // Extract templates from the response's variant
    // For now, we'll use placeholder values - in production, we'd need to extract from the actual variant
    let system_template = ""; // TODO: Extract from variant
    let user_template: Option<String> = None;
    let assistant_template: Option<String> = None;

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
    let output = serde_json::to_value(&eval_info.response).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize inference response: {e}"),
        })
    })?;

    // Get tags from datapoint (if any)
    let tags = match &eval_info.datapoint {
        tensorzero_core::endpoints::datasets::StoredDatapoint::Chat(dp) => dp.tags.clone(),
        tensorzero_core::endpoints::datasets::StoredDatapoint::Json(dp) => dp.tags.clone(),
    };

    // Get function name from datapoint
    let function_name = match &eval_info.datapoint {
        tensorzero_core::endpoints::datasets::StoredDatapoint::Chat(dp) => &dp.function_name,
        tensorzero_core::endpoints::datasets::StoredDatapoint::Json(dp) => &dp.function_name,
    };

    // Build the input object
    let input = serde_json::json!({
        "function_name": function_name,
        "model": eval_info.response.variant_name(),
        "system_template": system_template,
        "user_template": user_template,
        "assistant_template": assistant_template,
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

/// Parse the analysis response and extract the AnalysisReport
pub fn parse_analysis_response(response: &InferenceResponse) -> Result<AnalysisReport, Error> {
    let chat_response = match response {
        InferenceResponse::Chat(chat_response) => chat_response,
        InferenceResponse::Json(_) => {
            return Err(Error::new(ErrorDetails::Inference {
                message: "Expected Chat response from analyze function, got Json".to_string(),
            }))
        }
    };

    // Look for tool calls in the content blocks
    for block in &chat_response.content {
        if let ContentBlockChatOutput::ToolCall(tool_call) = block {
            // Parse the tool call into an AnalysisReport
            let args = tool_call.arguments.as_ref().ok_or_else(|| {
                Error::new(ErrorDetails::Inference {
                    message: "Tool call arguments not validated".to_string(),
                })
            })?;

            return match tool_call.name.as_deref() {
                Some("report_error") | Some("tensorzero::optimization::gepa::report_error") => {
                    Ok(serde_json::from_value(args.clone()).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!("Failed to parse report_error arguments: {e}"),
                        })
                    })?)
                }
                Some("report_improvement")
                | Some("tensorzero::optimization::gepa::report_improvement") => {
                    Ok(serde_json::from_value(args.clone()).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!("Failed to parse report_improvement arguments: {e}"),
                        })
                    })?)
                }
                Some("report_optimal") | Some("tensorzero::optimization::gepa::report_optimal") => {
                    Ok(serde_json::from_value(args.clone()).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!("Failed to parse report_optimal arguments: {e}"),
                        })
                    })?)
                }
                _ => Err(Error::new(ErrorDetails::Inference {
                    message: format!("Unknown analysis tool: {:?}", tool_call.name),
                })),
            };
        }
    }

    // If we didn't find a tool call, check if there's text content and warn
    for block in &chat_response.content {
        if let ContentBlockChatOutput::Text(_) = block {
            tracing::warn!(
                "Analyze function returned text instead of tool call for inference {}",
                chat_response.inference_id
            );
            // Return a default error report indicating the issue
            return Ok(AnalysisReport::Error {
                reasoning: "Analyzer returned text instead of using a tool call".to_string(),
                error_identification: "No structured analysis available".to_string(),
                root_cause_analysis: "The analysis model did not use the required tool calls"
                    .to_string(),
                correct_output: "Unknown".to_string(),
                key_insight: "Analysis infrastructure issue".to_string(),
            });
        }
    }

    Err(Error::new(ErrorDetails::Inference {
        message: "No tool call or text found in analyze response".to_string(),
    }))
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
        let input_data = build_analyze_input(eval_info, function_config)?;

        // Create ClientInferenceParams for the analyze function
        let params = ClientInferenceParams {
            function_name: Some("tensorzero::optimization::gepa::analyze".to_string()),
            model_name: None,
            episode_id: None,
            input: ClientInput {
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: serde_json::to_string(&input_data).map_err(|e| {
                            Error::new(ErrorDetails::Serialization {
                                message: format!("Failed to serialize analyze input: {e}"),
                            })
                        })?,
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
