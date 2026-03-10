//! Serde types for AWS Bedrock Converse API
//!
//! These types mirror the AWS Bedrock Converse API request/response structures
//! for direct HTTP calls without the AWS SDK.

use serde::{Deserialize, Serialize};

// =============================================================================
// Request Types
// =============================================================================

/// Request body for the Bedrock Converse API
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConverseRequest {
    /// The conversation messages
    pub messages: Vec<Message>,
    /// System prompts
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<SystemContentBlock>>,
    /// Inference configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_config: Option<InferenceConfig>,
    /// Tool configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    /// Additional model-specific fields (e.g., thinking configuration)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub additional_model_request_fields: Option<AdditionalModelRequestFields>,
}

/// Additional model-specific request fields
#[derive(Debug, Serialize)]
pub struct AdditionalModelRequestFields {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

/// Thinking/extended reasoning configuration
#[derive(Debug, Serialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: ThinkingType,
    pub budget_tokens: i32,
}

/// Thinking type
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingType {
    Enabled,
}

/// A conversation message
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

/// Message role
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// Content block in a message
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ContentBlock {
    Text(TextBlock),
    Image(ImageBlock),
    Document(DocumentBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    ReasoningContent(ReasoningContentBlock),
}

/// Text content block
#[derive(Debug, Serialize)]
pub struct TextBlock {
    pub text: String,
}

/// Image content block
#[derive(Debug, Serialize)]
pub struct ImageBlock {
    pub image: ImageSource,
}

/// Image source
#[derive(Debug, Serialize)]
pub struct ImageSource {
    pub format: String,
    pub source: ImageSourceData,
}

/// Image source data
#[derive(Debug, Serialize)]
pub struct ImageSourceData {
    pub bytes: String, // base64 encoded
}

/// Document content block
#[derive(Debug, Serialize)]
pub struct DocumentBlock {
    pub document: DocumentSource,
}

/// Document source
#[derive(Debug, Serialize)]
pub struct DocumentSource {
    pub format: String,
    pub name: String,
    pub source: DocumentSourceData,
}

/// Document source data
#[derive(Debug, Serialize)]
pub struct DocumentSourceData {
    pub bytes: String, // base64 encoded
}

/// Tool use content block (assistant calling a tool)
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolUseBlock {
    pub tool_use: ToolUseData,
}

/// Tool use data
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolUseData {
    pub tool_use_id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Tool result content block (user providing tool result)
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultBlock {
    pub tool_result: ToolResultData,
}

/// Tool result data
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultData {
    pub tool_use_id: String,
    pub content: Vec<ToolResultContent>,
}

/// Tool result content
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text { text: String },
}

/// Reasoning/thinking content block
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningContentBlock {
    pub reasoning_content: ReasoningContent,
}

/// Reasoning content variants
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ReasoningContent {
    ReasoningText(ReasoningText),
}

/// Reasoning text
#[derive(Debug, Serialize)]
pub struct ReasoningText {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// System content block
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum SystemContentBlock {
    Text { text: String },
}

/// Inference configuration
#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

/// Tool configuration
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

/// Tool definition
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub tool_spec: ToolSpec,
}

/// Tool specification
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: ToolInputSchema,
}

/// Tool input schema
#[derive(Debug, Serialize)]
pub struct ToolInputSchema {
    pub json: serde_json::Value,
}

/// Tool choice
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    Auto(AutoToolChoice),
    Any(AnyToolChoice),
    Tool(SpecificToolChoice),
}

/// Auto tool choice
#[derive(Debug, Default, Serialize)]
pub struct AutoToolChoice {}

/// Any tool choice (required)
#[derive(Debug, Default, Serialize)]
pub struct AnyToolChoice {}

/// Specific tool choice
#[derive(Debug, Serialize)]
pub struct SpecificToolChoice {
    pub name: String,
}

// =============================================================================
// Response Types (Non-streaming)
// =============================================================================

/// Response from the Bedrock Converse API
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConverseResponse {
    pub output: ConverseOutput,
    pub stop_reason: StopReason,
    pub usage: Usage,
    #[serde(default)]
    pub metrics: Option<Metrics>,
}

/// Converse output
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConverseOutput {
    pub message: Option<ResponseMessage>,
}

/// Response message
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResponseMessage {
    pub role: String,
    pub content: Vec<ResponseContentBlock>,
}

/// Response content block
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ResponseContentBlock {
    Text(String),
    #[serde(rename_all = "camelCase")]
    ToolUse {
        tool_use_id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename_all = "camelCase")]
    ReasoningContent(ResponseReasoningContent),
}

/// Response reasoning content
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ResponseReasoningContent {
    ReasoningText {
        text: String,
        #[serde(default)]
        signature: Option<String>,
    },
    RedactedContent(serde_json::Value),
}

/// Stop reason
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    ContentFiltered,
    GuardrailIntervened,
    #[serde(other)]
    Unknown,
}

/// Token usage
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub input_tokens: i32,
    pub output_tokens: i32,
    #[serde(default)]
    pub total_tokens: Option<i32>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<i32>,
    #[serde(default)]
    pub cache_write_input_tokens: Option<i32>,
}

/// Response metrics
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Metrics {
    pub latency_ms: Option<i64>,
}

// =============================================================================
// Streaming Event Types
// =============================================================================

/// Message start event
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageStartEvent {
    pub role: String,
    // p field is padding, ignored
}

/// Content block start event
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentBlockStartEvent {
    pub content_block_index: i32,
    #[serde(default)]
    pub start: Option<ContentBlockStart>,
}

/// Content block start variants
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ContentBlockStart {
    #[serde(rename_all = "camelCase")]
    ToolUse { tool_use_id: String, name: String },
}

/// Content block delta event
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentBlockDeltaEvent {
    pub content_block_index: i32,
    #[serde(default)]
    pub delta: Option<ContentBlockDelta>,
}

/// Content block delta variants
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ContentBlockDelta {
    Text(String),
    #[serde(rename_all = "camelCase")]
    ToolUse {
        input: String,
    },
    #[serde(rename_all = "camelCase")]
    ReasoningContent(ReasoningDelta),
}

/// Reasoning delta
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ReasoningDelta {
    Text(String),
    Signature(String),
    RedactedContent(serde_json::Value),
}

/// Content block stop event
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentBlockStopEvent {
    pub content_block_index: i32,
}

/// Message stop event
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageStopEvent {
    pub stop_reason: StopReason,
}

/// Metadata event (contains usage info)
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MetadataEvent {
    pub usage: Usage,
    #[serde(default)]
    pub metrics: Option<Metrics>,
}
