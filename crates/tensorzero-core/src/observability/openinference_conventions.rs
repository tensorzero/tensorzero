//! Emitters for OpenInference semantic-conventions span attributes that use
//! flattened indexed keys (e.g. `llm.input_messages.{i}.message.role`), which
//! the OpenTelemetry attribute model can't express as a single composite value.
//!
//! Spec: <https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md>
//!
//! Scalar attributes (`llm.system`, `llm.model_name`, `llm.token_count.*`,
//! `input.value`, `output.value`) are set directly at their call sites in
//! `crate::model` and `crate::config`; this module only handles the indexed
//! input-messages shape.
//!
//! The mapping mirrors the GenAI conventions module: `RequestMessage`s are
//! translated to OpenInference roles (with `User` promoted to `tool` when all
//! parts are tool results), text/thought content is concatenated into a single
//! `message.content` string, and tool calls are emitted as indexed sub-keys.
//! File and `Unknown` content blocks are skipped (no plain-string slot exists
//! for them in the spec).

use tensorzero_inference_types::{ContentBlock, RequestMessage};
use tensorzero_types::Role;
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

/// Sets `llm.input_messages.{i}.message.*` attributes for each request
/// message (and an optional leading `system` message). Caller is responsible
/// for gating on `format = OpenInference`.
pub fn apply_input_messages(span: &Span, system: Option<&str>, messages: &[RequestMessage]) {
    for (key, value) in build_input_message_attributes(system, messages) {
        span.set_attribute(key, value);
    }
}

/// Returns the flat list of `(key, value)` pairs that `apply_input_messages`
/// would set on a span. Exposed for unit testing.
fn build_input_message_attributes(
    system: Option<&str>,
    messages: &[RequestMessage],
) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut index = 0usize;

    if let Some(system_text) = system {
        push(&mut out, index, "message.role", "system");
        push(&mut out, index, "message.content", system_text);
        index += 1;
    }

    for msg in messages {
        let is_tool_role = matches!(msg.role, Role::User)
            && !msg.content.is_empty()
            && msg
                .content
                .iter()
                .all(|b| matches!(b, ContentBlock::ToolResult(_)));

        let role = if is_tool_role {
            "tool"
        } else {
            match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            }
        };
        push(&mut out, index, "message.role", role);

        let mut content = String::new();
        let mut tool_calls: Vec<(&str, &str, &str)> = Vec::new();
        let mut tool_call_id: Option<&str> = None;

        for block in &msg.content {
            match block {
                ContentBlock::Text(t) => append_paragraph(&mut content, &t.text),
                ContentBlock::Thought(thought) => {
                    if let Some(text) = thought.text.as_deref() {
                        append_paragraph(&mut content, text);
                    }
                }
                ContentBlock::ToolCall(tc) => {
                    tool_calls.push((tc.id.as_str(), tc.name.as_str(), tc.arguments.as_str()));
                }
                ContentBlock::ToolResult(tr) => {
                    append_paragraph(&mut content, &tr.result);
                    if tool_call_id.is_none() {
                        tool_call_id = Some(tr.id.as_str());
                    }
                }
                ContentBlock::File(_) | ContentBlock::Unknown(_) => {}
            }
        }

        if !content.is_empty() {
            push(&mut out, index, "message.content", &content);
        }
        if is_tool_role && let Some(id) = tool_call_id {
            push(&mut out, index, "message.tool_call_id", id);
        }
        for (j, (id, name, args)) in tool_calls.iter().enumerate() {
            let prefix = format!("message.tool_calls.{j}.tool_call");
            push(&mut out, index, &format!("{prefix}.id"), id);
            push(&mut out, index, &format!("{prefix}.function.name"), name);
            push(
                &mut out,
                index,
                &format!("{prefix}.function.arguments"),
                args,
            );
        }

        index += 1;
    }

    out
}

fn push(out: &mut Vec<(String, String)>, index: usize, suffix: &str, value: &str) {
    out.push((
        format!("llm.input_messages.{index}.{suffix}"),
        value.to_string(),
    ));
}

fn append_paragraph(buf: &mut String, text: &str) {
    if !buf.is_empty() {
        buf.push('\n');
    }
    buf.push_str(text);
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use googletest::prelude::*;
    use tensorzero_inference_types::{ContentBlock, RequestMessage};
    use tensorzero_types::{Role, Text, Thought, ToolCall, ToolResult};

    use super::build_input_message_attributes;

    fn into_map(pairs: Vec<(String, String)>) -> HashMap<String, String> {
        pairs.into_iter().collect()
    }

    #[gtest]
    fn system_prepended_at_index_zero() {
        let attrs = into_map(build_input_message_attributes(
            Some("be helpful"),
            &[RequestMessage {
                role: Role::User,
                content: vec![ContentBlock::Text(Text {
                    text: "hi".to_string(),
                })],
            }],
        ));
        expect_that!(
            attrs.get("llm.input_messages.0.message.role"),
            some(eq("system"))
        );
        expect_that!(
            attrs.get("llm.input_messages.0.message.content"),
            some(eq("be helpful"))
        );
        expect_that!(
            attrs.get("llm.input_messages.1.message.role"),
            some(eq("user"))
        );
        expect_that!(
            attrs.get("llm.input_messages.1.message.content"),
            some(eq("hi"))
        );
    }

    #[gtest]
    fn no_system_starts_at_index_zero() {
        let attrs = into_map(build_input_message_attributes(
            None,
            &[RequestMessage {
                role: Role::Assistant,
                content: vec![ContentBlock::Text(Text {
                    text: "hello".to_string(),
                })],
            }],
        ));
        expect_that!(
            attrs.get("llm.input_messages.0.message.role"),
            some(eq("assistant"))
        );
        expect_that!(
            attrs.get("llm.input_messages.0.message.content"),
            some(eq("hello"))
        );
    }

    #[gtest]
    fn tool_call_emitted_as_indexed_subkeys() {
        let attrs = into_map(build_input_message_attributes(
            None,
            &[RequestMessage {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: r#"{"city":"Paris"}"#.to_string(),
                })],
            }],
        ));
        expect_that!(
            attrs.get("llm.input_messages.0.message.role"),
            some(eq("assistant"))
        );
        // No text content means `.content` is omitted.
        expect_that!(attrs.get("llm.input_messages.0.message.content"), none());
        expect_that!(
            attrs.get("llm.input_messages.0.message.tool_calls.0.tool_call.id"),
            some(eq("call_1"))
        );
        expect_that!(
            attrs.get("llm.input_messages.0.message.tool_calls.0.tool_call.function.name"),
            some(eq("get_weather"))
        );
        expect_that!(
            attrs.get("llm.input_messages.0.message.tool_calls.0.tool_call.function.arguments"),
            some(eq(r#"{"city":"Paris"}"#))
        );
    }

    #[gtest]
    fn tool_result_promotes_role_and_emits_tool_call_id() {
        let attrs = into_map(build_input_message_attributes(
            None,
            &[RequestMessage {
                role: Role::User,
                content: vec![ContentBlock::ToolResult(ToolResult {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    result: r#"{"temp":57}"#.to_string(),
                })],
            }],
        ));
        expect_that!(
            attrs.get("llm.input_messages.0.message.role"),
            some(eq("tool"))
        );
        expect_that!(
            attrs.get("llm.input_messages.0.message.content"),
            some(eq(r#"{"temp":57}"#))
        );
        expect_that!(
            attrs.get("llm.input_messages.0.message.tool_call_id"),
            some(eq("call_1"))
        );
    }

    #[gtest]
    fn mixed_text_and_tool_result_keeps_user_role() {
        let attrs = into_map(build_input_message_attributes(
            None,
            &[RequestMessage {
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
            }],
        ));
        // Not all blocks are tool results → role stays `user`, tool_call_id
        // is not emitted.
        expect_that!(
            attrs.get("llm.input_messages.0.message.role"),
            some(eq("user"))
        );
        expect_that!(
            attrs.get("llm.input_messages.0.message.tool_call_id"),
            none()
        );
        // Text + result are concatenated.
        expect_that!(
            attrs.get("llm.input_messages.0.message.content"),
            some(eq("context\nok"))
        );
    }

    #[gtest]
    fn thought_text_is_concatenated_into_content() {
        let attrs = into_map(build_input_message_attributes(
            None,
            &[RequestMessage {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thought(Thought {
                        text: Some("thinking...".to_string()),
                        signature: None,
                        summary: None,
                        provider_type: None,
                        extra_data: None,
                    }),
                    ContentBlock::Text(Text {
                        text: "answer".to_string(),
                    }),
                ],
            }],
        ));
        expect_that!(
            attrs.get("llm.input_messages.0.message.content"),
            some(eq("thinking...\nanswer"))
        );
    }
}
