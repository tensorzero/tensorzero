//! Serialization of TensorZero inference types into the OpenTelemetry GenAI
//! semantic-conventions content-capture shape.
//!
//! We emit four JSON-string span attributes on the `model_provider_inference`
//! span when `include_content = true` and `format = opentelemetry`:
//!
//! - `gen_ai.input.messages`
//! - `gen_ai.output.messages`
//! - `gen_ai.system_instructions`
//! - `gen_ai.tool.definitions`
//!
//! Spec: <https://opentelemetry.io/docs/specs/semconv/gen-ai/>
//!
//! This module is intentionally free of any OpenTelemetry SDK or `tracing`
//! dependency. It maps TensorZero types into `Serialize`-derived structs whose
//! JSON shape matches the spec one-to-one, so the wire contract is visible in
//! the type definitions alone.
//!
//! Callers do `serde_json::to_string(&value).unwrap_or_default()` to produce
//! the final span attribute string.

use mime::MediaType;
use serde::Serialize;
use serde_json::Value;
use tensorzero_inference_types::{
    ContentBlock, ContentBlockChunk, ContentBlockOutput, FunctionToolDef, LazyFile,
    OpenAICustomTool, OpenAICustomToolFormat, ProviderInferenceResponseChunk,
    ProviderToolCallConfig, RequestMessage,
};
use tensorzero_types::{FinishReason, Role};

/// Message role as defined by the GenAI semantic conventions.
///
/// TensorZero's internal [`Role`] only has `User` and `Assistant`. We promote
/// a user-role message to `Tool` when all its parts are tool-call responses,
/// matching the role most observability backends expect (Phoenix, Langfuse,
/// Jaeger).
#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GenAiRole {
    User,
    Assistant,
    Tool,
    System,
}

/// The coarse media category of a [`GenAiPart::Blob`], [`GenAiPart::Uri`], or
/// [`GenAiPart::File`]. Derived from the leading `type` segment of the MIME
/// type.
#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GenAiModality {
    Text,
    Image,
    Audio,
    Video,
    Other,
}

/// A single part of a message, per the GenAI spec. `#[serde(tag = "type")]`
/// gives the `{"type": "<variant>", ...fields}` shape the spec mandates.
#[derive(Debug, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GenAiPart {
    Text {
        content: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
    ToolCallResponse {
        id: String,
        response: Value,
    },
    Reasoning {
        content: String,
    },
    /// Inline binary payload (base64 encoded).
    Blob {
        #[serde(skip_serializing_if = "Option::is_none")]
        modality: Option<GenAiModality>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        content: String,
    },
    /// Remote file reference by URL.
    Uri {
        #[serde(skip_serializing_if = "Option::is_none")]
        modality: Option<GenAiModality>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        uri: String,
    },
    /// Pre-uploaded file reference by opaque ID.
    File {
        #[serde(skip_serializing_if = "Option::is_none")]
        modality: Option<GenAiModality>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        file_id: String,
    },
    /// Opaque fall-through for TensorZero-specific content that doesn't fit
    /// the spec (e.g. `ContentBlock::Unknown`, storage errors).
    #[serde(rename = "tensorzero.unknown")]
    TensorzeroUnknown {
        content: Value,
    },
}

/// A single message in the input or output array.
#[derive(Debug, Serialize, PartialEq)]
pub struct GenAiMessage {
    pub role: GenAiRole,
    pub parts: Vec<GenAiPart>,
    /// Only populated on assistant (output) messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// A tool definition passed to the model, for `gen_ai.tool.definitions`.
///
/// `Custom` mirrors OpenAI's `type: "custom"` tool kind (free-form or
/// grammar-constrained text output) so telemetry preserves the distinction
/// between function tools and custom tools as they are sent to the provider.
#[derive(Debug, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GenAiToolDefinition {
    Function {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        parameters: Value,
    },
    Custom {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        format: Option<OpenAICustomToolFormat>,
    },
}

// =============================================================================
// Converters
// =============================================================================

pub fn to_genai_messages(messages: &[RequestMessage]) -> Vec<GenAiMessage> {
    messages.iter().map(convert_request_message).collect()
}

pub fn to_genai_output(
    output: &[ContentBlockOutput],
    finish_reason: Option<FinishReason>,
) -> Vec<GenAiMessage> {
    let parts = output.iter().map(convert_output_block).collect();
    vec![GenAiMessage {
        role: GenAiRole::Assistant,
        parts,
        finish_reason: finish_reason.map(finish_reason_to_string),
    }]
}

/// Aggregate a buffered stream of provider response chunks into the GenAI
/// output messages shape, for use when setting `gen_ai.output.messages` on a
/// streaming `model_provider_inference` span.
///
/// Chunks sharing an `id` collapse into a single part (text appended, tool-call
/// raw arguments concatenated then JSON-parsed once, thought text appended).
/// The last non-`None` `finish_reason` across chunks is used.
pub fn to_genai_output_from_chunks(chunks: &[ProviderInferenceResponseChunk]) -> Vec<GenAiMessage> {
    #[derive(Default)]
    struct PartialToolCall {
        name: String,
        raw_arguments: String,
    }

    enum Acc {
        Text(String),
        ToolCall(PartialToolCall),
        Reasoning(String),
        Unknown(Value),
    }

    // Keyed by `(PartKind, id)` so that providers which reuse the same `id`
    // across different chunk kinds (e.g. a text chunk and a thought chunk both
    // named `"0"`) accumulate into separate parts instead of colliding.
    #[derive(Clone, Copy, Eq, Hash, PartialEq)]
    enum PartKind {
        Text,
        ToolCall,
        Reasoning,
        Unknown,
    }

    type Key = (PartKind, String);

    let mut order: Vec<Key> = Vec::new();
    let mut accs: std::collections::HashMap<Key, Acc> = std::collections::HashMap::new();
    let mut finish_reason: Option<FinishReason> = None;

    for chunk in chunks {
        if let Some(fr) = chunk.finish_reason {
            finish_reason = Some(fr);
        }
        for block in &chunk.content {
            match block {
                ContentBlockChunk::Text(t) => {
                    let key = (PartKind::Text, t.id.clone());
                    match accs.get_mut(&key) {
                        Some(Acc::Text(s)) => s.push_str(&t.text),
                        Some(_) => {}
                        None => {
                            order.push(key.clone());
                            accs.insert(key, Acc::Text(t.text.clone()));
                        }
                    }
                }
                ContentBlockChunk::ToolCall(tc) => {
                    let key = (PartKind::ToolCall, tc.id.clone());
                    match accs.get_mut(&key) {
                        Some(Acc::ToolCall(partial)) => {
                            if let Some(name) = &tc.raw_name
                                && partial.name.is_empty()
                            {
                                partial.name.clone_from(name);
                            }
                            partial.raw_arguments.push_str(&tc.raw_arguments);
                        }
                        Some(_) => {}
                        None => {
                            order.push(key.clone());
                            accs.insert(
                                key,
                                Acc::ToolCall(PartialToolCall {
                                    name: tc.raw_name.clone().unwrap_or_default(),
                                    raw_arguments: tc.raw_arguments.clone(),
                                }),
                            );
                        }
                    }
                }
                ContentBlockChunk::Thought(t) => {
                    let fragment = t.text.clone().unwrap_or_default();
                    let key = (PartKind::Reasoning, t.id.clone());
                    match accs.get_mut(&key) {
                        Some(Acc::Reasoning(s)) => s.push_str(&fragment),
                        Some(_) => {}
                        None => {
                            order.push(key.clone());
                            accs.insert(key, Acc::Reasoning(fragment));
                        }
                    }
                }
                ContentBlockChunk::Unknown(u) => {
                    let key = (PartKind::Unknown, u.id.clone());
                    if let std::collections::hash_map::Entry::Vacant(slot) = accs.entry(key.clone())
                    {
                        order.push(key);
                        slot.insert(Acc::Unknown(u.data.clone()));
                    }
                }
            }
        }
    }

    let parts: Vec<GenAiPart> = order
        .into_iter()
        .filter_map(|key| accs.remove(&key).map(|acc| (key, acc)))
        .map(|((_, id), acc)| match acc {
            Acc::Text(content) => GenAiPart::Text { content },
            Acc::ToolCall(partial) => GenAiPart::ToolCall {
                id,
                name: partial.name,
                arguments: parse_json_or_string(&partial.raw_arguments),
            },
            Acc::Reasoning(content) => GenAiPart::Reasoning { content },
            Acc::Unknown(content) => GenAiPart::TensorzeroUnknown { content },
        })
        .collect();

    vec![GenAiMessage {
        role: GenAiRole::Assistant,
        parts,
        finish_reason: finish_reason.map(finish_reason_to_string),
    }]
}

pub fn to_genai_system_instructions(system: Option<&str>) -> Option<Vec<GenAiPart>> {
    let text = system?;
    Some(vec![GenAiPart::Text {
        content: text.to_owned(),
    }])
}

pub fn to_genai_tool_definitions(
    tool_config: Option<&ProviderToolCallConfig>,
) -> Option<Vec<GenAiToolDefinition>> {
    let config = tool_config?;
    let mut defs: Vec<GenAiToolDefinition> = Vec::new();
    for tool in &config.tools {
        defs.push(function_tool_to_definition(tool));
    }
    for tool in &config.openai_custom_tools {
        defs.push(openai_custom_tool_to_definition(tool));
    }
    if defs.is_empty() { None } else { Some(defs) }
}

// =============================================================================
// Internals
// =============================================================================

fn convert_request_message(message: &RequestMessage) -> GenAiMessage {
    let parts: Vec<GenAiPart> = message.content.iter().map(convert_input_block).collect();
    let role = if matches!(message.role, Role::User)
        && !parts.is_empty()
        && parts
            .iter()
            .all(|p| matches!(p, GenAiPart::ToolCallResponse { .. }))
    {
        GenAiRole::Tool
    } else {
        match message.role {
            Role::User => GenAiRole::User,
            Role::Assistant => GenAiRole::Assistant,
        }
    };
    GenAiMessage {
        role,
        parts,
        finish_reason: None,
    }
}

fn convert_input_block(block: &ContentBlock) -> GenAiPart {
    match block {
        ContentBlock::Text(text) => GenAiPart::Text {
            content: text.text.clone(),
        },
        ContentBlock::ToolCall(call) => GenAiPart::ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: parse_json_or_string(&call.arguments),
        },
        ContentBlock::ToolResult(result) => GenAiPart::ToolCallResponse {
            id: result.id.clone(),
            response: parse_json_or_string(&result.result),
        },
        ContentBlock::File(file) => convert_lazy_file(file),
        ContentBlock::Thought(thought) => GenAiPart::Reasoning {
            content: thought.text.clone().unwrap_or_default(),
        },
        ContentBlock::Unknown(unknown) => GenAiPart::TensorzeroUnknown {
            content: unknown.data.clone(),
        },
    }
}

fn convert_output_block(block: &ContentBlockOutput) -> GenAiPart {
    match block {
        ContentBlockOutput::Text(text) => GenAiPart::Text {
            content: text.text.clone(),
        },
        ContentBlockOutput::ToolCall(call) => GenAiPart::ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: parse_json_or_string(&call.arguments),
        },
        ContentBlockOutput::Thought(thought) => GenAiPart::Reasoning {
            content: thought.text.clone().unwrap_or_default(),
        },
        ContentBlockOutput::Unknown(unknown) => GenAiPart::TensorzeroUnknown {
            content: unknown.data.clone(),
        },
    }
}

fn convert_lazy_file(file: &LazyFile) -> GenAiPart {
    match file {
        LazyFile::Url { file_url, .. } => GenAiPart::Uri {
            modality: Some(
                file_url
                    .mime_type
                    .as_ref()
                    .map_or(GenAiModality::Other, modality_for),
            ),
            mime_type: file_url.mime_type.as_ref().map(|m| m.to_string()),
            uri: file_url.url.to_string(),
        },
        LazyFile::Base64(pending) => GenAiPart::Blob {
            modality: Some(modality_for(&pending.0.file.mime_type)),
            mime_type: Some(pending.0.file.mime_type.to_string()),
            content: pending.0.data.clone(),
        },
        LazyFile::ObjectStoragePointer {
            metadata,
            storage_path,
            ..
        } => {
            if let Some(url) = &metadata.source_url {
                GenAiPart::Uri {
                    modality: Some(modality_for(&metadata.mime_type)),
                    mime_type: Some(metadata.mime_type.to_string()),
                    uri: url.to_string(),
                }
            } else {
                GenAiPart::File {
                    modality: Some(modality_for(&metadata.mime_type)),
                    mime_type: Some(metadata.mime_type.to_string()),
                    file_id: storage_path.path.to_string(),
                }
            }
        }
        LazyFile::ObjectStorage(resolved) => GenAiPart::Blob {
            modality: Some(modality_for(&resolved.file.mime_type)),
            mime_type: Some(resolved.file.mime_type.to_string()),
            content: resolved.data.clone(),
        },
    }
}

fn modality_for(mime_type: &MediaType) -> GenAiModality {
    match mime_type.type_() {
        mime::IMAGE => GenAiModality::Image,
        mime::AUDIO => GenAiModality::Audio,
        mime::VIDEO => GenAiModality::Video,
        mime::TEXT => GenAiModality::Text,
        _ => GenAiModality::Other,
    }
}

fn parse_json_or_string(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned()))
}

fn finish_reason_to_string(reason: FinishReason) -> String {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::StopSequence => "stop_sequence",
        FinishReason::Length => "length",
        FinishReason::ToolCall => "tool_call",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::Unknown => "unknown",
    }
    .to_owned()
}

fn function_tool_to_definition(tool: &FunctionToolDef) -> GenAiToolDefinition {
    GenAiToolDefinition::Function {
        name: tool.name.clone(),
        description: non_empty(&tool.description),
        parameters: tool.parameters.clone(),
    }
}

fn openai_custom_tool_to_definition(tool: &OpenAICustomTool) -> GenAiToolDefinition {
    GenAiToolDefinition::Custom {
        name: tool.name.clone(),
        description: tool.description.clone(),
        format: tool.format.clone(),
    }
}

fn non_empty(s: &str) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use futures::FutureExt;
    use googletest::prelude::*;
    use googletest_matchers::matches_json_literal;
    use serde_json::{Value, json};
    use std::time::Duration;
    use tensorzero_inference_types::{
        AllowedTools, AllowedToolsChoice, ContentBlock, ContentBlockOutput, FileUrl,
        FunctionToolDef, LazyFile, OpenAICustomTool, PendingObjectStoreFile, ProviderTool,
        ProviderToolCallConfig, ProviderToolScope, RequestMessage, TextChunk, ThoughtChunk,
        ToolCallChunk,
    };
    use tensorzero_types::{
        Base64FileMetadata, FinishReason, ObjectStorageFile, ObjectStoragePointer, Role,
        StorageKind, StoragePath, Text, Thought, ToolCall, ToolChoice, ToolResult, Unknown,
    };
    use url::Url;

    use super::*;

    fn to_json(value: &impl Serialize) -> Value {
        serde_json::to_value(value).expect("serialization should succeed")
    }

    fn png_mime() -> mime::MediaType {
        mime::IMAGE_PNG
    }

    fn wav_mime() -> mime::MediaType {
        "audio/wav".parse().expect("valid mime")
    }

    fn pdf_mime() -> mime::MediaType {
        mime::APPLICATION_PDF
    }

    fn make_storage_path(path: &str) -> StoragePath {
        StoragePath {
            kind: StorageKind::Filesystem {
                path: "/tmp".to_string(),
            },
            path: object_store::path::Path::from(path),
        }
    }

    fn make_object_storage_file(mime: mime::MediaType, data: &str) -> ObjectStorageFile {
        ObjectStorageFile {
            file: ObjectStoragePointer {
                source_url: None,
                mime_type: mime,
                storage_path: make_storage_path("objects/test.bin"),
                detail: None,
                filename: None,
            },
            data: data.to_string(),
        }
    }

    // ── Messages & parts ────────────────────────────────────────────────

    #[gtest]
    fn text_block_serializes_as_text_part() {
        let msg = RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::Text(Text {
                text: "hello".to_string(),
            })],
        };
        let out = to_genai_messages(&[msg]);
        expect_that!(
            to_json(&out),
            matches_json_literal!([
                {
                    "role": "user",
                    "parts": [{ "type": "text", "content": "hello" }]
                }
            ])
        );
    }

    #[gtest]
    fn tool_call_parses_json_arguments() {
        let msg = RequestMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                arguments: r#"{"city":"Paris"}"#.to_string(),
            })],
        };
        expect_that!(
            to_json(&to_genai_messages(&[msg])),
            matches_json_literal!([
                {
                    "role": "assistant",
                    "parts": [{
                        "type": "tool_call",
                        "id": "call_1",
                        "name": "get_weather",
                        "arguments": {"city": "Paris"}
                    }]
                }
            ])
        );
    }

    #[gtest]
    fn tool_call_falls_back_to_string_for_invalid_json() {
        let msg = RequestMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "echo".to_string(),
                arguments: "not json {".to_string(),
            })],
        };
        expect_that!(
            to_json(&to_genai_messages(&[msg])),
            matches_json_literal!([
                {
                    "role": "assistant",
                    "parts": [{
                        "type": "tool_call",
                        "id": "call_1",
                        "name": "echo",
                        "arguments": "not json {"
                    }]
                }
            ])
        );
    }

    #[gtest]
    fn tool_result_promotes_role_to_tool() {
        let msg = RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                result: r#"{"temp":57}"#.to_string(),
            })],
        };
        expect_that!(
            to_json(&to_genai_messages(&[msg])),
            matches_json_literal!([
                {
                    "role": "tool",
                    "parts": [{
                        "type": "tool_call_response",
                        "id": "call_1",
                        "response": {"temp": 57}
                    }]
                }
            ])
        );
    }

    #[gtest]
    fn user_role_preserved_when_mixed_with_tool_result() {
        let msg = RequestMessage {
            role: Role::User,
            content: vec![
                ContentBlock::Text(Text {
                    text: "context".to_string(),
                }),
                ContentBlock::ToolResult(ToolResult {
                    id: "call_1".to_string(),
                    name: "foo".to_string(),
                    result: "ok".to_string(),
                }),
            ],
        };
        let value = to_json(&to_genai_messages(&[msg]));
        expect_that!(value.pointer("/0/role"), some(eq(&json!("user"))));
    }

    #[gtest]
    fn thought_with_text_maps_to_reasoning() {
        let msg = RequestMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::Thought(Thought {
                text: Some("thinking...".to_string()),
                signature: None,
                summary: None,
                provider_type: None,
                extra_data: None,
            })],
        };
        expect_that!(
            to_json(&to_genai_messages(&[msg])),
            matches_json_literal!([
                {
                    "role": "assistant",
                    "parts": [{"type": "reasoning", "content": "thinking..."}]
                }
            ])
        );
    }

    #[gtest]
    fn thought_without_text_uses_empty_content() {
        let msg = RequestMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::Thought(Thought {
                text: None,
                signature: None,
                summary: None,
                provider_type: None,
                extra_data: None,
            })],
        };
        expect_that!(
            to_json(&to_genai_messages(&[msg])),
            matches_json_literal!([
                {
                    "role": "assistant",
                    "parts": [{"type": "reasoning", "content": ""}]
                }
            ])
        );
    }

    #[gtest]
    fn unknown_block_maps_to_tensorzero_unknown() {
        let msg = RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::Unknown(Unknown {
                data: json!({"custom": "payload"}),
                model_name: None,
                provider_name: None,
            })],
        };
        expect_that!(
            to_json(&to_genai_messages(&[msg])),
            matches_json_literal!([
                {
                    "role": "user",
                    "parts": [{
                        "type": "tensorzero.unknown",
                        "content": {"custom": "payload"}
                    }]
                }
            ])
        );
    }

    // ── Files ───────────────────────────────────────────────────────────

    #[gtest]
    fn url_file_maps_to_uri_part_with_modality() {
        let lazy = LazyFile::Url {
            file_url: FileUrl {
                url: Url::parse("https://example.com/img.png").unwrap(),
                mime_type: Some(png_mime()),
                detail: None,
                filename: None,
            },
            future: async { panic!("should not resolve") }.boxed().shared(),
        };
        let part = convert_lazy_file(&lazy);
        expect_that!(
            to_json(&part),
            matches_json_literal!({
                "type": "uri",
                "modality": "image",
                "mime_type": "image/png",
                "uri": "https://example.com/img.png"
            })
        );
    }

    #[gtest]
    fn url_file_without_mime_falls_back_to_other_modality() {
        let lazy = LazyFile::Url {
            file_url: FileUrl {
                url: Url::parse("https://example.com/data").unwrap(),
                mime_type: None,
                detail: None,
                filename: None,
            },
            future: async { panic!("should not resolve") }.boxed().shared(),
        };
        expect_that!(
            to_json(&convert_lazy_file(&lazy)),
            matches_json_literal!({
                "type": "uri",
                "modality": "other",
                "uri": "https://example.com/data"
            })
        );
    }

    #[gtest]
    fn base64_file_maps_to_blob_part() {
        let pending_inner = ObjectStorageFile {
            file: ObjectStoragePointer {
                source_url: None,
                mime_type: wav_mime(),
                storage_path: make_storage_path("inline/audio"),
                detail: None,
                filename: None,
            },
            data: "ZmFrZS1hdWRpbw==".to_string(),
        };
        let lazy = LazyFile::Base64(PendingObjectStoreFile(pending_inner));
        expect_that!(
            to_json(&convert_lazy_file(&lazy)),
            matches_json_literal!({
                "type": "blob",
                "modality": "audio",
                "mime_type": "audio/wav",
                "content": "ZmFrZS1hdWRpbw=="
            })
        );
    }

    #[gtest]
    fn object_storage_pointer_without_source_url_maps_to_file_part() {
        let lazy = LazyFile::ObjectStoragePointer {
            metadata: Base64FileMetadata {
                source_url: None,
                mime_type: png_mime(),
                detail: None,
                filename: None,
            },
            storage_path: make_storage_path("uploads/abc123"),
            future: async { panic!("should not resolve") }.boxed().shared(),
        };
        expect_that!(
            to_json(&convert_lazy_file(&lazy)),
            matches_json_literal!({
                "type": "file",
                "modality": "image",
                "mime_type": "image/png",
                "file_id": "uploads/abc123"
            })
        );
    }

    #[gtest]
    fn object_storage_pointer_with_source_url_maps_to_uri_part() {
        let lazy = LazyFile::ObjectStoragePointer {
            metadata: Base64FileMetadata {
                source_url: Some(Url::parse("s3://bucket/key").unwrap()),
                mime_type: pdf_mime(),
                detail: None,
                filename: None,
            },
            storage_path: make_storage_path("uploads/abc123"),
            future: async { panic!("should not resolve") }.boxed().shared(),
        };
        expect_that!(
            to_json(&convert_lazy_file(&lazy)),
            matches_json_literal!({
                "type": "uri",
                "modality": "other",
                "mime_type": "application/pdf",
                "uri": "s3://bucket/key"
            })
        );
    }

    #[gtest]
    fn resolved_object_storage_maps_to_blob_part() {
        let resolved = make_object_storage_file(png_mime(), "YmFzZTY0");
        let lazy = LazyFile::ObjectStorage(resolved);
        expect_that!(
            to_json(&convert_lazy_file(&lazy)),
            matches_json_literal!({
                "type": "blob",
                "modality": "image",
                "mime_type": "image/png",
                "content": "YmFzZTY0"
            })
        );
    }

    // ── Modality derivation ─────────────────────────────────────────────

    #[gtest]
    fn modality_for_image() {
        expect_that!(modality_for(&png_mime()), eq(&GenAiModality::Image));
    }

    #[gtest]
    fn modality_for_audio() {
        expect_that!(modality_for(&wav_mime()), eq(&GenAiModality::Audio));
    }

    #[gtest]
    fn modality_for_video() {
        let mime: mime::MediaType = "video/mp4".parse().unwrap();
        expect_that!(modality_for(&mime), eq(&GenAiModality::Video));
    }

    #[gtest]
    fn modality_for_text() {
        let mime: mime::MediaType = "text/plain".parse().unwrap();
        expect_that!(modality_for(&mime), eq(&GenAiModality::Text));
    }

    #[gtest]
    fn modality_for_pdf_is_other() {
        expect_that!(modality_for(&pdf_mime()), eq(&GenAiModality::Other));
    }

    // ── Output messages ─────────────────────────────────────────────────

    #[gtest]
    fn output_messages_single_text_with_finish_reason() {
        let blocks = vec![ContentBlockOutput::Text(Text {
            text: "done".to_string(),
        })];
        expect_that!(
            to_json(&to_genai_output(&blocks, Some(FinishReason::Stop))),
            matches_json_literal!([
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": "done"}],
                    "finish_reason": "stop"
                }
            ])
        );
    }

    #[gtest]
    fn output_messages_omits_finish_reason_when_absent() {
        let blocks = vec![ContentBlockOutput::Text(Text {
            text: "...".to_string(),
        })];
        let value = to_json(&to_genai_output(&blocks, None));
        expect_that!(value.pointer("/0/finish_reason"), none());
    }

    #[gtest]
    fn output_finish_reason_mapping_covers_all_variants() {
        let cases = [
            (FinishReason::Stop, "stop"),
            (FinishReason::StopSequence, "stop_sequence"),
            (FinishReason::Length, "length"),
            (FinishReason::ToolCall, "tool_call"),
            (FinishReason::ContentFilter, "content_filter"),
            (FinishReason::Unknown, "unknown"),
        ];
        for (reason, expected) in cases {
            let out = to_genai_output(&[], Some(reason));
            expect_that!(out[0].finish_reason.as_deref(), eq(Some(expected)));
        }
    }

    #[gtest]
    fn output_tool_call_serializes_like_input() {
        let blocks = vec![ContentBlockOutput::ToolCall(ToolCall {
            id: "c1".to_string(),
            name: "f".to_string(),
            arguments: r#"{"x":1}"#.to_string(),
        })];
        expect_that!(
            to_json(&to_genai_output(&blocks, Some(FinishReason::ToolCall))),
            matches_json_literal!([
                {
                    "role": "assistant",
                    "parts": [{
                        "type": "tool_call",
                        "id": "c1",
                        "name": "f",
                        "arguments": {"x": 1}
                    }],
                    "finish_reason": "tool_call"
                }
            ])
        );
    }

    // ── System instructions ─────────────────────────────────────────────

    #[gtest]
    fn system_instructions_some() {
        expect_that!(
            to_json(&to_genai_system_instructions(Some("be helpful"))),
            matches_json_literal!([{"type": "text", "content": "be helpful"}])
        );
    }

    #[gtest]
    fn system_instructions_none() {
        expect_that!(to_genai_system_instructions(None), none());
    }

    // ── Tool definitions ────────────────────────────────────────────────

    fn empty_provider_tool_config() -> ProviderToolCallConfig {
        ProviderToolCallConfig {
            tools: Vec::new(),
            provider_tools: Vec::new(),
            openai_custom_tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools {
                tools: Vec::new(),
                choice: AllowedToolsChoice::FunctionDefault,
            },
        }
    }

    #[gtest]
    fn tool_definitions_none_when_no_config() {
        expect_that!(to_genai_tool_definitions(None), none());
    }

    #[gtest]
    fn tool_definitions_none_when_empty() {
        let config = empty_provider_tool_config();
        expect_that!(to_genai_tool_definitions(Some(&config)), none());
    }

    #[gtest]
    fn tool_definitions_function_tool() {
        let mut config = empty_provider_tool_config();
        config.tools.push(FunctionToolDef {
            name: "get_weather".to_string(),
            description: "Get current weather".to_string(),
            parameters: json!({"type": "object"}),
            strict: false,
        });
        expect_that!(
            to_json(&to_genai_tool_definitions(Some(&config))),
            matches_json_literal!([
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object"}
                }
            ])
        );
    }

    #[gtest]
    fn tool_definitions_function_with_empty_description_and_openai_custom() {
        let mut config = empty_provider_tool_config();
        config.tools.push(FunctionToolDef {
            name: "dyn".to_string(),
            description: String::new(),
            parameters: json!({}),
            strict: false,
        });
        config.openai_custom_tools.push(OpenAICustomTool {
            name: "custom".to_string(),
            description: Some("an openai custom tool".to_string()),
            format: None,
        });
        let value = to_json(&to_genai_tool_definitions(Some(&config)));
        // Function tool: empty description omitted.
        expect_that!(
            value.pointer("/0"),
            some(eq(&json!({
                "type": "function",
                "name": "dyn",
                "parameters": {}
            })))
        );
        // OpenAI custom tool: preserved as `custom` kind.
        expect_that!(
            value.pointer("/1"),
            some(eq(&json!({
                "type": "custom",
                "name": "custom",
                "description": "an openai custom tool"
            })))
        );
    }

    // ── Streaming aggregation ───────────────────────────────────────────

    fn make_chunk(
        content: Vec<ContentBlockChunk>,
        finish_reason: Option<FinishReason>,
    ) -> ProviderInferenceResponseChunk {
        ProviderInferenceResponseChunk::new(
            content,
            None,
            String::new(),
            Duration::ZERO,
            finish_reason,
        )
    }

    #[gtest]
    fn chunks_text_concatenation() {
        let chunks = vec![
            make_chunk(
                vec![ContentBlockChunk::Text(TextChunk {
                    id: "t1".to_string(),
                    text: "Hello ".to_string(),
                })],
                None,
            ),
            make_chunk(
                vec![ContentBlockChunk::Text(TextChunk {
                    id: "t1".to_string(),
                    text: "world".to_string(),
                })],
                Some(FinishReason::Stop),
            ),
        ];
        expect_that!(
            to_json(&to_genai_output_from_chunks(&chunks)),
            matches_json_literal!([
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": "Hello world"}],
                    "finish_reason": "stop"
                }
            ])
        );
    }

    #[gtest]
    fn chunks_tool_call_raw_arguments_concatenate_and_parse() {
        let chunks = vec![
            make_chunk(
                vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: "tc1".to_string(),
                    raw_name: Some("get_weather".to_string()),
                    raw_arguments: r#"{"loc"#.to_string(),
                })],
                None,
            ),
            make_chunk(
                vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: "tc1".to_string(),
                    raw_name: None,
                    raw_arguments: r#"ation":"Paris"}"#.to_string(),
                })],
                Some(FinishReason::ToolCall),
            ),
        ];
        expect_that!(
            to_json(&to_genai_output_from_chunks(&chunks)),
            matches_json_literal!([
                {
                    "role": "assistant",
                    "parts": [{
                        "type": "tool_call",
                        "id": "tc1",
                        "name": "get_weather",
                        "arguments": {"location": "Paris"}
                    }],
                    "finish_reason": "tool_call"
                }
            ])
        );
    }

    #[gtest]
    fn chunks_thought_concatenation() {
        let chunks = vec![
            make_chunk(
                vec![ContentBlockChunk::Thought(ThoughtChunk {
                    id: "r1".to_string(),
                    text: Some("step one. ".to_string()),
                    signature: None,
                    summary_id: None,
                    summary_text: None,
                    provider_type: None,
                    extra_data: None,
                })],
                None,
            ),
            make_chunk(
                vec![ContentBlockChunk::Thought(ThoughtChunk {
                    id: "r1".to_string(),
                    text: Some("step two.".to_string()),
                    signature: None,
                    summary_id: None,
                    summary_text: None,
                    provider_type: None,
                    extra_data: None,
                })],
                None,
            ),
        ];
        let value = to_json(&to_genai_output_from_chunks(&chunks));
        expect_that!(
            value.pointer("/0/parts/0"),
            some(eq(
                &json!({"type": "reasoning", "content": "step one. step two."})
            ))
        );
    }

    #[gtest]
    fn chunks_preserve_first_seen_order_across_ids() {
        let chunks = vec![make_chunk(
            vec![
                ContentBlockChunk::Text(TextChunk {
                    id: "t1".to_string(),
                    text: "first".to_string(),
                }),
                ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: "tc1".to_string(),
                    raw_name: Some("f".to_string()),
                    raw_arguments: "{}".to_string(),
                }),
            ],
            Some(FinishReason::ToolCall),
        )];
        let value = to_json(&to_genai_output_from_chunks(&chunks));
        expect_that!(value.pointer("/0/parts/0/type"), some(eq(&json!("text"))));
        expect_that!(
            value.pointer("/0/parts/1/type"),
            some(eq(&json!("tool_call")))
        );
    }

    #[gtest]
    fn chunks_last_non_none_finish_reason_wins() {
        let chunks = vec![
            make_chunk(
                vec![ContentBlockChunk::Text(TextChunk {
                    id: "t1".to_string(),
                    text: "x".to_string(),
                })],
                None,
            ),
            make_chunk(vec![], Some(FinishReason::Length)),
            make_chunk(vec![], None),
        ];
        let value = to_json(&to_genai_output_from_chunks(&chunks));
        expect_that!(
            value.pointer("/0/finish_reason"),
            some(eq(&json!("length")))
        );
    }

    #[gtest]
    fn chunks_same_id_across_kinds_accumulate_separately() {
        // A provider that reuses positional ids across chunk kinds (here a
        // `Text` chunk and a `Thought` chunk both named `"0"`) should produce
        // two independent parts, not one dropped and one kept.
        let chunks = vec![
            make_chunk(
                vec![ContentBlockChunk::Text(TextChunk {
                    id: "0".to_string(),
                    text: "visible".to_string(),
                })],
                None,
            ),
            make_chunk(
                vec![ContentBlockChunk::Thought(ThoughtChunk {
                    id: "0".to_string(),
                    text: Some("thinking".to_string()),
                    signature: None,
                    summary_id: None,
                    summary_text: None,
                    provider_type: None,
                    extra_data: None,
                })],
                None,
            ),
        ];
        expect_that!(
            to_json(&to_genai_output_from_chunks(&chunks)),
            matches_json_literal!([
                {
                    "role": "assistant",
                    "parts": [
                        {"type": "text", "content": "visible"},
                        {"type": "reasoning", "content": "thinking"}
                    ]
                }
            ])
        );
    }

    #[gtest]
    fn tool_definitions_ignores_provider_tools() {
        let mut config = empty_provider_tool_config();
        config.provider_tools.push(ProviderTool {
            scope: ProviderToolScope::Unscoped,
            tool: json!({"type": "code_interpreter"}),
        });
        expect_that!(to_genai_tool_definitions(Some(&config)), none());
    }
}
