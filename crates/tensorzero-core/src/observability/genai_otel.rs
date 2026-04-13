//! Conversion between the OpenTelemetry GenAI semantic conventions and
//! TensorZero's internal inference representation.
//!
//! Spec references:
//! - Spans: <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/>
//! - Events: <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/>
//!
//! This module is intentionally side-effect-free and self-contained. It does
//! not call any TensorZero APIs; it just parses an OTel-shaped representation
//! into a TensorZero-shaped one. Useful for ingesting traces from other
//! systems, replaying captured spans, and (eventually) building bridges to
//! other observability tools.

use std::collections::HashMap;

use opentelemetry::KeyValue;
use tensorzero_inference_types::{FinishReason, ProviderInferenceResponse};
use tensorzero_types::Role;
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::inference::types::ModelInferenceRequest;

/// A logical OpenTelemetry GenAI span as we receive it: a flat attribute map
/// plus an ordered list of named events with their own attributes.
///
/// This is intentionally untyped to tolerate slightly older or non-standard
/// emitters — every field is optional and unknowns are ignored.
#[derive(Debug, Default, Clone)]
#[non_exhaustive]
pub struct GenAiSpan {
    /// Span attributes such as `gen_ai.request.model`, `gen_ai.system`, etc.
    pub attributes: HashMap<String, String>,
    /// Ordered span events such as `gen_ai.user.message`, `gen_ai.choice`.
    pub events: Vec<GenAiEvent>,
}

/// A single OTel span event with its attributes.
#[derive(Debug, Clone)]
pub struct GenAiEvent {
    pub name: String,
    pub attributes: HashMap<String, String>,
}

/// Parsed inference reconstructed from an OTel GenAI span.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenAiInference {
    /// `gen_ai.system` — provider/system name (e.g. "openai", "anthropic").
    pub system: Option<String>,
    /// `gen_ai.request.model` — requested model.
    pub request_model: Option<String>,
    /// `gen_ai.response.model` — actual model that responded (if different).
    pub response_model: Option<String>,
    /// `gen_ai.response.id` — provider-side id of the response.
    pub response_id: Option<String>,
    /// `gen_ai.operation.name` — typically "chat" or "embeddings".
    pub operation: Option<String>,
    /// Request parameters parsed from `gen_ai.request.*`.
    pub params: GenAiRequestParams,
    /// Token usage parsed from `gen_ai.usage.*`.
    pub usage: GenAiUsage,
    /// Messages parsed from `gen_ai.{role}.message` events.
    pub messages: Vec<GenAiMessage>,
    /// Choices parsed from `gen_ai.choice` events.
    pub choices: Vec<GenAiChoice>,
    /// `gen_ai.response.finish_reasons` — first finish reason if present.
    pub finish_reason: Option<String>,
    /// `error.type` — error class if the call failed.
    pub error_type: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenAiRequestParams {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u32>,
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenAiUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

/// One logical input message reconstructed from a `gen_ai.{role}.message` event.
#[derive(Debug, Clone, PartialEq)]
pub struct GenAiMessage {
    pub role: GenAiRole,
    /// Raw event content. We don't try to parse JSON here — callers can.
    pub content: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenAiRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenAiChoice {
    /// Per the OTel GenAI spec, choice index is an unsigned integer.
    pub index: Option<u32>,
    pub finish_reason: Option<String>,
    pub message: Option<String>,
}

impl GenAiRole {
    /// Bridge to TensorZero's `Role`. Returns `None` for `System` and `Tool`
    /// since TensorZero doesn't model those as message roles (system prompts
    /// live on the request itself; tool messages are routed differently).
    #[must_use]
    pub fn to_tensorzero_role(&self) -> Option<tensorzero_types::Role> {
        match self {
            Self::User => Some(tensorzero_types::Role::User),
            Self::Assistant => Some(tensorzero_types::Role::Assistant),
            Self::System | Self::Tool => None,
        }
    }
}

/// Parse an OTel GenAI span into a `GenAiInference`.
///
/// Unknown attributes and events are silently ignored. Malformed numeric
/// values (e.g. a non-integer `gen_ai.request.max_tokens`) are dropped
/// rather than causing an error — observability data is best-effort.
pub fn parse_genai_span(span: &GenAiSpan) -> GenAiInference {
    let attrs = &span.attributes;
    let mut out = GenAiInference {
        system: attrs.get("gen_ai.system").cloned(),
        request_model: attrs.get("gen_ai.request.model").cloned(),
        response_model: attrs.get("gen_ai.response.model").cloned(),
        response_id: attrs.get("gen_ai.response.id").cloned(),
        operation: attrs.get("gen_ai.operation.name").cloned(),
        error_type: attrs.get("error.type").cloned(),
        ..Default::default()
    };

    out.params = GenAiRequestParams {
        max_tokens: attrs
            .get("gen_ai.request.max_tokens")
            .and_then(|v| v.parse().ok()),
        temperature: attrs
            .get("gen_ai.request.temperature")
            .and_then(|v| v.parse().ok()),
        top_p: attrs
            .get("gen_ai.request.top_p")
            .and_then(|v| v.parse().ok()),
        frequency_penalty: attrs
            .get("gen_ai.request.frequency_penalty")
            .and_then(|v| v.parse().ok()),
        presence_penalty: attrs
            .get("gen_ai.request.presence_penalty")
            .and_then(|v| v.parse().ok()),
        seed: attrs
            .get("gen_ai.request.seed")
            .and_then(|v| v.parse().ok()),
        stop_sequences: parse_csv_or_array(attrs.get("gen_ai.request.stop_sequences")),
    };

    out.usage = GenAiUsage {
        input_tokens: attrs
            .get("gen_ai.usage.input_tokens")
            .and_then(|v| v.parse().ok()),
        output_tokens: attrs
            .get("gen_ai.usage.output_tokens")
            .and_then(|v| v.parse().ok()),
        total_tokens: attrs
            .get("gen_ai.usage.total_tokens")
            .and_then(|v| v.parse().ok()),
    };

    let finish_reasons = parse_csv_or_array(attrs.get("gen_ai.response.finish_reasons"));
    out.finish_reason = finish_reasons.into_iter().next();

    for event in &span.events {
        if let Some(role) = role_from_event_name(&event.name) {
            out.messages.push(GenAiMessage {
                role,
                content: event.attributes.get("content").cloned(),
            });
        } else if event.name == "gen_ai.choice" {
            out.choices.push(GenAiChoice {
                index: event.attributes.get("index").and_then(|v| v.parse().ok()),
                finish_reason: event.attributes.get("finish_reason").cloned(),
                message: event.attributes.get("message").cloned(),
            });
        }
    }

    // If the span didn't have a response.finish_reasons attribute but did
    // have a choice with a finish reason, surface it on the top-level field.
    if out.finish_reason.is_none() {
        out.finish_reason = out.choices.iter().find_map(|c| c.finish_reason.clone());
    }

    out
}

fn role_from_event_name(name: &str) -> Option<GenAiRole> {
    match name {
        "gen_ai.system.message" => Some(GenAiRole::System),
        "gen_ai.user.message" => Some(GenAiRole::User),
        "gen_ai.assistant.message" => Some(GenAiRole::Assistant),
        "gen_ai.tool.message" => Some(GenAiRole::Tool),
        _ => None,
    }
}

/// Stop sequences and finish reasons may arrive as either a JSON array (when
/// the emitter sent an OTel array attribute) or a comma-separated string
/// (some collectors flatten arrays this way).
///
/// **Caveat:** the CSV fallback is lossy — a sequence containing a literal
/// comma (e.g. `"1, 2, 3"`) will be split incorrectly. JSON array form is
/// preferred and tried first.
fn parse_csv_or_array(value: Option<&String>) -> Vec<String> {
    let Some(raw) = value else { return vec![] };
    let trimmed = raw.trim();
    match serde_json::from_str::<Vec<String>>(trimmed) {
        Ok(parsed) => parsed,
        Err(e) => {
            tracing::trace!(
                "parse_csv_or_array: JSON array parse failed ({e}); falling back to CSV split"
            );
            trimmed
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        }
    }
}

// =============================================================================
// Emission helpers (TensorZero → OTel GenAI)
// =============================================================================

/// Map a TensorZero `FinishReason` to its OTel GenAI standard string.
/// https://opentelemetry.io/docs/specs/semconv/attributes-registry/gen-ai/
pub fn finish_reason_to_otel_str(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::StopSequence => "stop_sequence",
        FinishReason::Length => "length",
        FinishReason::ToolCall => "tool_calls",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::Unknown => "error",
    }
}

/// Inverse of [`finish_reason_to_otel_str`]: parse an OTel GenAI finish-reason
/// string back into a `FinishReason`. Unknown values map to `FinishReason::Unknown`.
#[must_use]
pub fn parse_finish_reason(s: &str) -> FinishReason {
    match s {
        "stop" => FinishReason::Stop,
        "stop_sequence" => FinishReason::StopSequence,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCall,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Unknown,
    }
}

/// Emit `gen_ai.{role}.message` events for each input message on the current span.
/// https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/
pub fn emit_input_message_events(span: &Span, request: &ModelInferenceRequest<'_>) {
    if let Some(system) = &request.system {
        span.add_event(
            "gen_ai.system.message",
            vec![KeyValue::new("content", system.clone())],
        );
    }

    for message in &request.messages {
        let event_name = match message.role {
            Role::User => "gen_ai.user.message",
            Role::Assistant => "gen_ai.assistant.message",
        };
        let content_json = serde_json::to_string(&message.content)
            .unwrap_or_else(|_| "<serialization error>".to_string());
        span.add_event(event_name, vec![KeyValue::new("content", content_json)]);
    }
}

/// Emit a `gen_ai.choice` event with the response content on the current span.
/// https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/
pub fn emit_choice_event(span: &Span, response: &ProviderInferenceResponse) {
    let content_json = serde_json::to_string(&response.output)
        .unwrap_or_else(|_| "<serialization error>".to_string());
    let mut attributes = vec![
        KeyValue::new("index", 0_i64),
        KeyValue::new("message", content_json),
    ];
    if let Some(finish_reason) = response.finish_reason {
        attributes.push(KeyValue::new(
            "finish_reason",
            finish_reason_to_otel_str(finish_reason),
        ));
    }
    span.add_event("gen_ai.choice", attributes);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_span() -> GenAiSpan {
        GenAiSpan::default()
    }

    fn attr(span: &mut GenAiSpan, key: &str, value: &str) {
        span.attributes.insert(key.to_string(), value.to_string());
    }

    fn event(span: &mut GenAiSpan, name: &str, attrs: &[(&str, &str)]) {
        let attributes = attrs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        span.events.push(GenAiEvent {
            name: name.to_string(),
            attributes,
        });
    }

    #[test]
    fn parses_minimal_span() {
        let mut span = make_span();
        attr(&mut span, "gen_ai.system", "openai");
        attr(&mut span, "gen_ai.request.model", "gpt-4o");
        attr(&mut span, "gen_ai.operation.name", "chat");

        let parsed = parse_genai_span(&span);
        assert_eq!(parsed.system.as_deref(), Some("openai"));
        assert_eq!(parsed.request_model.as_deref(), Some("gpt-4o"));
        assert_eq!(parsed.operation.as_deref(), Some("chat"));
        assert_eq!(parsed.params, GenAiRequestParams::default());
        assert!(parsed.messages.is_empty());
    }

    #[test]
    fn parses_full_request_params() {
        let mut span = make_span();
        attr(&mut span, "gen_ai.request.max_tokens", "1024");
        attr(&mut span, "gen_ai.request.temperature", "0.7");
        attr(&mut span, "gen_ai.request.top_p", "0.9");
        attr(&mut span, "gen_ai.request.frequency_penalty", "0.1");
        attr(&mut span, "gen_ai.request.presence_penalty", "-0.1");
        attr(&mut span, "gen_ai.request.seed", "42");

        let parsed = parse_genai_span(&span);
        assert_eq!(parsed.params.max_tokens, Some(1024));
        assert!((parsed.params.temperature.unwrap() - 0.7).abs() < 1e-6);
        assert!((parsed.params.top_p.unwrap() - 0.9).abs() < 1e-6);
        assert!((parsed.params.frequency_penalty.unwrap() - 0.1).abs() < 1e-6);
        assert!((parsed.params.presence_penalty.unwrap() + 0.1).abs() < 1e-6);
        assert_eq!(parsed.params.seed, Some(42));
    }

    #[test]
    fn malformed_numbers_are_dropped_silently() {
        let mut span = make_span();
        attr(&mut span, "gen_ai.request.max_tokens", "not-a-number");
        attr(&mut span, "gen_ai.request.seed", "");
        let parsed = parse_genai_span(&span);
        assert_eq!(parsed.params.max_tokens, None);
        assert_eq!(parsed.params.seed, None);
    }

    #[test]
    fn parses_stop_sequences_as_json_array() {
        let mut span = make_span();
        attr(
            &mut span,
            "gen_ai.request.stop_sequences",
            r#"["END", "STOP"]"#,
        );
        let parsed = parse_genai_span(&span);
        assert_eq!(parsed.params.stop_sequences, vec!["END", "STOP"]);
    }

    #[test]
    fn parses_stop_sequences_as_csv_fallback() {
        let mut span = make_span();
        attr(&mut span, "gen_ai.request.stop_sequences", "END, STOP, ");
        let parsed = parse_genai_span(&span);
        assert_eq!(parsed.params.stop_sequences, vec!["END", "STOP"]);
    }

    #[test]
    fn parses_usage() {
        let mut span = make_span();
        attr(&mut span, "gen_ai.usage.input_tokens", "100");
        attr(&mut span, "gen_ai.usage.output_tokens", "50");
        attr(&mut span, "gen_ai.usage.total_tokens", "150");
        let parsed = parse_genai_span(&span);
        assert_eq!(parsed.usage.input_tokens, Some(100));
        assert_eq!(parsed.usage.output_tokens, Some(50));
        assert_eq!(parsed.usage.total_tokens, Some(150));
    }

    #[test]
    fn parses_message_events() {
        let mut span = make_span();
        event(
            &mut span,
            "gen_ai.system.message",
            &[("content", "You are a helpful assistant.")],
        );
        event(
            &mut span,
            "gen_ai.user.message",
            &[("content", "Hello there.")],
        );
        event(
            &mut span,
            "gen_ai.assistant.message",
            &[("content", "Hi! How can I help?")],
        );
        event(&mut span, "gen_ai.tool.message", &[("content", "{}")]);

        let parsed = parse_genai_span(&span);
        assert_eq!(parsed.messages.len(), 4);
        assert_eq!(parsed.messages[0].role, GenAiRole::System);
        assert_eq!(parsed.messages[1].role, GenAiRole::User);
        assert_eq!(parsed.messages[2].role, GenAiRole::Assistant);
        assert_eq!(parsed.messages[3].role, GenAiRole::Tool);
        assert_eq!(parsed.messages[1].content.as_deref(), Some("Hello there."));
    }

    #[test]
    fn parses_choice_events() {
        let mut span = make_span();
        event(
            &mut span,
            "gen_ai.choice",
            &[
                ("index", "0"),
                ("finish_reason", "stop"),
                ("message", r#"{"content":"42"}"#),
            ],
        );
        let parsed = parse_genai_span(&span);
        assert_eq!(parsed.choices.len(), 1);
        assert_eq!(parsed.choices[0].index, Some(0));
        assert_eq!(parsed.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(parsed.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn finish_reason_attribute_takes_precedence_over_choice() {
        let mut span = make_span();
        attr(&mut span, "gen_ai.response.finish_reasons", "[\"length\"]");
        event(
            &mut span,
            "gen_ai.choice",
            &[("finish_reason", "stop"), ("message", "{}")],
        );
        let parsed = parse_genai_span(&span);
        assert_eq!(parsed.finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn unknown_events_and_attributes_are_ignored() {
        let mut span = make_span();
        attr(&mut span, "some.unknown.attr", "ignored");
        event(&mut span, "gen_ai.unknown.event", &[("content", "x")]);
        event(&mut span, "random_event", &[]);

        let parsed = parse_genai_span(&span);
        assert!(parsed.messages.is_empty());
        assert!(parsed.choices.is_empty());
    }

    #[test]
    fn parses_error_type() {
        let mut span = make_span();
        attr(&mut span, "error.type", "InferenceServer");
        let parsed = parse_genai_span(&span);
        assert_eq!(parsed.error_type.as_deref(), Some("InferenceServer"));
    }

    #[test]
    fn end_to_end_example() {
        // A full happy-path span as an emitter (e.g. a TensorZero gateway in
        // OpenTelemetry mode) might produce.
        let mut span = make_span();
        attr(&mut span, "gen_ai.system", "anthropic");
        attr(&mut span, "gen_ai.operation.name", "chat");
        attr(&mut span, "gen_ai.request.model", "claude-sonnet-4-5");
        attr(&mut span, "gen_ai.request.max_tokens", "256");
        attr(&mut span, "gen_ai.request.temperature", "0.2");
        attr(&mut span, "gen_ai.usage.input_tokens", "12");
        attr(&mut span, "gen_ai.usage.output_tokens", "5");
        attr(&mut span, "gen_ai.response.id", "resp_abc123");
        attr(&mut span, "gen_ai.response.finish_reasons", "[\"stop\"]");
        event(&mut span, "gen_ai.user.message", &[("content", "Ping?")]);
        event(
            &mut span,
            "gen_ai.choice",
            &[
                ("index", "0"),
                ("finish_reason", "stop"),
                ("message", "Pong."),
            ],
        );

        let parsed = parse_genai_span(&span);
        assert_eq!(parsed.system.as_deref(), Some("anthropic"));
        assert_eq!(parsed.request_model.as_deref(), Some("claude-sonnet-4-5"));
        assert_eq!(parsed.params.max_tokens, Some(256));
        assert_eq!(parsed.usage.input_tokens, Some(12));
        assert_eq!(parsed.usage.output_tokens, Some(5));
        assert_eq!(parsed.response_id.as_deref(), Some("resp_abc123"));
        assert_eq!(parsed.messages.len(), 1);
        assert_eq!(parsed.choices.len(), 1);
        assert_eq!(parsed.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn finish_reason_round_trips() {
        for reason in [
            FinishReason::Stop,
            FinishReason::StopSequence,
            FinishReason::Length,
            FinishReason::ToolCall,
            FinishReason::ContentFilter,
            FinishReason::Unknown,
        ] {
            let s = finish_reason_to_otel_str(reason);
            let back = parse_finish_reason(s);
            // Unknown round-trips to itself via the "error" string going
            // back to Unknown — that's intentional.
            if matches!(reason, FinishReason::Unknown) {
                assert_eq!(back, FinishReason::Unknown);
            } else {
                assert_eq!(back, reason, "round-trip failed for {reason:?}");
            }
        }
    }

    #[test]
    fn parse_finish_reason_handles_unknown_strings() {
        assert_eq!(
            parse_finish_reason("totally_made_up"),
            FinishReason::Unknown
        );
        assert_eq!(parse_finish_reason(""), FinishReason::Unknown);
    }

    #[test]
    fn genai_role_bridges_to_tensorzero_role() {
        assert_eq!(
            GenAiRole::User.to_tensorzero_role(),
            Some(tensorzero_types::Role::User)
        );
        assert_eq!(
            GenAiRole::Assistant.to_tensorzero_role(),
            Some(tensorzero_types::Role::Assistant)
        );
        // System and Tool don't have a TensorZero `Role` equivalent.
        assert_eq!(GenAiRole::System.to_tensorzero_role(), None);
        assert_eq!(GenAiRole::Tool.to_tensorzero_role(), None);
    }
}
