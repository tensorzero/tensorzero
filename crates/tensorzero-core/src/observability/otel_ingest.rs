//! Parse OTLP JSON traces (the standard wire format for OTel GenAI spans)
//! and convert them into TensorZero's `StoredModelInference` type.
//!
//! This module handles the full OTLP JSON envelope: `resourceSpans` →
//! `scopeSpans` → `spans`, with typed attribute values (`stringValue`,
//! `intValue`, `boolValue`, `arrayValue`).
//!
//! Spec references:
//! - OTLP JSON: <https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding>
//! - GenAI spans: <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/>

// TODO: remove once the OTel ingestion endpoint is wired up.
#![expect(
    dead_code,
    reason = "only called from tests until ingestion endpoint lands"
)]

use serde::Deserialize;
use uuid::Uuid;

use crate::inference::types::StoredModelInference;

// =============================================================================
// OTLP JSON wire types (subset needed for GenAI span parsing)
// =============================================================================
//
// These are modeled as structs with optional fields rather than enums because
// the OTLP JSON protobuf encoding sends typed values as optional fields on a
// single JSON object (not a tagged union), so serde's untagged enum would give
// poor error messages and worse perf.

/// Top-level OTLP export request (one JSONL line).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OtlpExportRequest {
    #[serde(default)]
    pub resource_spans: Vec<ResourceSpans>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ResourceSpans {
    #[serde(default)]
    pub scope_spans: Vec<ScopeSpans>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ScopeSpans {
    #[serde(default)]
    pub spans: Vec<OtlpSpan>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OtlpSpan {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub start_time_unix_nano: String,
    #[serde(default)]
    pub end_time_unix_nano: String,
    #[serde(default)]
    pub attributes: Vec<OtlpAttribute>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OtlpAttribute {
    pub key: String,
    pub value: OtlpValue,
}

/// OTLP typed attribute value. We handle the types that appear in GenAI spans.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OtlpValue {
    #[serde(default)]
    pub string_value: Option<String>,
    #[serde(default)]
    pub int_value: Option<serde_json::Value>, // can be string or number in OTLP JSON
    #[serde(default)]
    pub bool_value: Option<bool>,
    #[serde(default)]
    pub array_value: Option<OtlpArrayValue>,
}

#[derive(Debug, Deserialize, Clone)]
pub(crate) struct OtlpArrayValue {
    #[serde(default)]
    pub values: Vec<OtlpValue>,
}

impl OtlpValue {
    /// Borrow the string value without allocating (when the value is a `stringValue`).
    fn as_str(&self) -> Option<&str> {
        self.string_value.as_deref()
    }

    /// Extract as an owned string, coercing ints and bools.
    fn as_string(&self) -> Option<String> {
        if let Some(s) = &self.string_value {
            return Some(s.clone());
        }
        if let Some(v) = &self.int_value {
            return Some(match v {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                other => other.to_string(),
            });
        }
        if let Some(b) = self.bool_value {
            return Some(b.to_string());
        }
        None
    }

    /// Extract as u32 (for token counts, max_tokens, etc.).
    /// Returns `None` if the value overflows `u32`.
    fn as_u32(&self) -> Option<u32> {
        if let Some(v) = &self.int_value {
            return match v {
                serde_json::Value::String(s) => s.parse().ok(),
                serde_json::Value::Number(n) => n.as_u64().and_then(|n| u32::try_from(n).ok()),
                _ => None,
            };
        }
        if let Some(s) = &self.string_value {
            return s.parse().ok();
        }
        None
    }

    /// Extract the first string from an array value (for finish_reasons).
    fn first_array_str(&self) -> Option<&str> {
        self.array_value.as_ref()?.values.first()?.as_str()
    }
}

// =============================================================================
// Helpers: attribute lookup on a span
// =============================================================================

/// A thin wrapper for convenient attribute access on an OTLP span.
struct SpanAttrs<'a>(&'a [OtlpAttribute]);

impl<'a> SpanAttrs<'a> {
    fn get(&self, key: &str) -> Option<&'a OtlpValue> {
        self.0.iter().find(|a| a.key == key).map(|a| &a.value)
    }

    /// Return an owned `String` (allocates). Use for fields that go into
    /// `StoredModelInference` owned-string fields.
    fn string(&self, key: &str) -> Option<String> {
        self.get(key)?.as_string()
    }

    fn u32(&self, key: &str) -> Option<u32> {
        self.get(key)?.as_u32()
    }

    /// Borrow the first string element of an array attribute (no allocation).
    fn first_array_str(&self, key: &str) -> Option<&'a str> {
        self.get(key)?.first_array_str()
    }
}

// =============================================================================
// Conversion: OtlpSpan → StoredModelInference
// =============================================================================

/// Map an OTel GenAI finish-reason string to a `FinishReason`.
fn parse_finish_reason(s: &str) -> tensorzero_inference_types::FinishReason {
    use tensorzero_inference_types::FinishReason;
    match s {
        "stop" => FinishReason::Stop,
        "stop_sequence" => FinishReason::StopSequence,
        "length" => FinishReason::Length,
        "tool_calls" | "tool_call" => FinishReason::ToolCall,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Unknown,
    }
}

/// Convert a single OTLP GenAI span into a `StoredModelInference`.
///
/// Fields that don't have a natural source in the OTel span (e.g.
/// `inference_id`, `function_name`, `variant_name`) are left as defaults
/// — the caller is responsible for filling those in.
pub fn otlp_span_to_stored_model_inference(span: &OtlpSpan) -> StoredModelInference {
    let attrs = SpanAttrs(&span.attributes);

    // Timing: compute response_time_ms from start/end nanos.
    let start_ns: u128 = span.start_time_unix_nano.parse().unwrap_or(0);
    let end_ns: u128 = span.end_time_unix_nano.parse().unwrap_or(0);
    let response_time_ms = if end_ns > start_ns {
        let ms = (end_ns - start_ns) / 1_000_000;
        Some(u32::try_from(ms).unwrap_or(u32::MAX))
    } else {
        None
    };

    // Token usage
    let input_tokens = attrs.u32("gen_ai.usage.input_tokens");
    let output_tokens = attrs.u32("gen_ai.usage.output_tokens");

    // Finish reason: from the array attribute `gen_ai.response.finish_reasons`
    let finish_reason = attrs
        .first_array_str("gen_ai.response.finish_reasons")
        .map(parse_finish_reason);

    // The raw request/response aren't available from OTel traces in a standard
    // form — the closest we have is `gen_ai.input.messages` / `gen_ai.output.messages`
    // which are JSON strings representing the semantic content.
    let raw_request = attrs.string("gen_ai.input.messages");
    let raw_response = attrs.string("gen_ai.output.messages");

    StoredModelInference {
        id: Uuid::now_v7(),
        inference_id: Uuid::nil(),
        function_name: String::new(),
        variant_name: String::new(),
        raw_request,
        raw_response,
        system: None, // System prompt is embedded in input.messages, not a separate attribute
        input_messages: None, // Would need further parsing of the gen_ai.input.messages JSON
        output: None, // Would need further parsing of the gen_ai.output.messages JSON
        input_tokens,
        output_tokens,
        provider_cache_read_input_tokens: attrs.u32("gen_ai.usage.cache_read.input_tokens"),
        provider_cache_write_input_tokens: attrs.u32("gen_ai.usage.cache_creation.input_tokens"),
        response_time_ms,
        model_name: attrs.string("gen_ai.request.model").unwrap_or_default(),
        model_provider_name: attrs.string("gen_ai.provider.name").unwrap_or_default(),
        ttft_ms: None, // Not available in standard OTel GenAI spans
        cached: false,
        cost: None,
        finish_reason,
        snapshot_hash: None,
        provider_response_id: attrs.string("gen_ai.response.id"),
        response_model_name: attrs.string("gen_ai.response.model"),
        operation: attrs.string("gen_ai.operation.name"),
        timestamp: None,
    }
}

/// Parse an entire OTLP JSONL file (one `OtlpExportRequest` per line)
/// and return all `StoredModelInference` entries.
pub fn parse_otlp_jsonl(jsonl: &str) -> Result<Vec<StoredModelInference>, serde_json::Error> {
    let mut results = Vec::new();
    for line in jsonl.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let export: OtlpExportRequest = serde_json::from_str(line)?;
        for rs in &export.resource_spans {
            for ss in &rs.scope_spans {
                for span in &ss.spans {
                    results.push(otlp_span_to_stored_model_inference(span));
                }
            }
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;
    use tensorzero_inference_types::FinishReason;

    const OPENAI_CHAT: &str = include_str!("../../testdata/otel_traces/openai_chat.jsonl");
    const OPENAI_RESPONSES: &str =
        include_str!("../../testdata/otel_traces/openai_responses.jsonl");
    const ANTHROPIC_CHAT: &str = include_str!("../../testdata/otel_traces/anthropic_chat.jsonl");
    const ANTHROPIC_TOOL_USE: &str =
        include_str!("../../testdata/otel_traces/anthropic_tool_use.jsonl");

    /// Helper: parse and verify all required fields are populated on every span.
    fn assert_required_fields(results: &[StoredModelInference]) {
        for (i, mi) in results.iter().enumerate() {
            expect_that!(mi.model_name, not(eq("")), "span {i}: model_name");
            expect_that!(
                mi.model_provider_name,
                not(eq("")),
                "span {i}: model_provider_name"
            );
            expect_that!(
                mi.provider_response_id,
                some(anything()),
                "span {i}: provider_response_id"
            );
            expect_that!(
                mi.response_model_name,
                some(anything()),
                "span {i}: response_model_name"
            );
            expect_that!(mi.operation, some(anything()), "span {i}: operation");
            expect_that!(mi.input_tokens, some(anything()), "span {i}: input_tokens");
            expect_that!(
                mi.output_tokens,
                some(anything()),
                "span {i}: output_tokens"
            );
            expect_that!(
                mi.finish_reason,
                some(anything()),
                "span {i}: finish_reason"
            );
            expect_that!(
                mi.response_time_ms,
                some(anything()),
                "span {i}: response_time_ms"
            );
        }
    }

    // ── OpenAI Chat Completions ──────────────────────────────────────────

    #[gtest]
    fn openai_chat_parses_two_spans() {
        let results = parse_otlp_jsonl(OPENAI_CHAT).expect("should parse");
        expect_that!(results.len(), eq(2));
        assert_required_fields(&results);
    }

    #[gtest]
    fn openai_chat_first_span() {
        let results = parse_otlp_jsonl(OPENAI_CHAT).expect("should parse");
        let mi = &results[0];

        expect_that!(mi.model_name, eq("gpt-4o-mini"));
        expect_that!(mi.model_provider_name, eq("openai"));
        expect_that!(
            mi.provider_response_id.as_deref(),
            some(eq("chatcmpl-DVHHVlFx61AuQIq4gZvooyrpLQ0NL"))
        );
        expect_that!(
            mi.response_model_name.as_deref(),
            some(eq("gpt-4o-mini-2024-07-18"))
        );
        expect_that!(mi.operation.as_deref(), some(eq("chat")));
        expect_that!(mi.input_tokens, some(eq(13)));
        expect_that!(mi.output_tokens, some(eq(5)));
        expect_that!(mi.finish_reason, some(eq(FinishReason::Stop)));
        // (1776346969531762000 - 1776346968790316000) / 1_000_000 = 741 ms
        expect_that!(mi.response_time_ms, some(eq(741)));
        expect_that!(
            mi.raw_request.as_deref(),
            some(contains_substring("Say hello in three words"))
        );
        expect_that!(
            mi.raw_response.as_deref(),
            some(contains_substring("Hello there, friend!"))
        );
        expect_that!(mi.provider_cache_read_input_tokens, some(eq(0)));
    }

    #[gtest]
    fn openai_chat_second_span() {
        let results = parse_otlp_jsonl(OPENAI_CHAT).expect("should parse");
        let mi = &results[1];

        expect_that!(
            mi.provider_response_id.as_deref(),
            some(eq("chatcmpl-DVHHhqgjSU5zDs00XROzyYfGtgKv2"))
        );
        expect_that!(mi.input_tokens, some(eq(13)));
        expect_that!(mi.output_tokens, some(eq(6)));
        expect_that!(mi.finish_reason, some(eq(FinishReason::Stop)));
    }

    // ── OpenAI Responses API ─────────────────────────────────────────────

    #[gtest]
    fn openai_responses_parses_one_span() {
        let results = parse_otlp_jsonl(OPENAI_RESPONSES).expect("should parse");
        expect_that!(results.len(), eq(1));
        assert_required_fields(&results);
    }

    #[gtest]
    fn openai_responses_span() {
        let results = parse_otlp_jsonl(OPENAI_RESPONSES).expect("should parse");
        let mi = &results[0];

        expect_that!(mi.model_name, eq("gpt-4o-mini"));
        expect_that!(mi.model_provider_name, eq("openai"));
        expect_that!(
            mi.provider_response_id.as_deref(),
            some(eq(
                "resp_09146d0742be5c060169e0e761eb3c8190a5695efe2e8df0d6"
            ))
        );
        expect_that!(
            mi.response_model_name.as_deref(),
            some(eq("gpt-4o-mini-2024-07-18"))
        );
        expect_that!(mi.operation.as_deref(), some(eq("chat")));
        expect_that!(mi.input_tokens, some(eq(26)));
        expect_that!(mi.output_tokens, some(eq(7)));
        expect_that!(mi.finish_reason, some(eq(FinishReason::Stop)));
        expect_that!(
            mi.raw_request.as_deref(),
            some(contains_substring("You are a helpful assistant"))
        );
    }

    // ── Anthropic Chat ───────────────────────────────────────────────────

    #[gtest]
    fn anthropic_chat_parses_one_span() {
        let results = parse_otlp_jsonl(ANTHROPIC_CHAT).expect("should parse");
        expect_that!(results.len(), eq(1));
        assert_required_fields(&results);
    }

    #[gtest]
    fn anthropic_chat_span() {
        let results = parse_otlp_jsonl(ANTHROPIC_CHAT).expect("should parse");
        let mi = &results[0];

        expect_that!(mi.model_name, eq("claude-sonnet-4-20250514"));
        expect_that!(mi.model_provider_name, eq("anthropic"));
        expect_that!(
            mi.provider_response_id.as_deref(),
            some(eq("msg_014eJGsS2mHFBdvg71gToaaB"))
        );
        expect_that!(
            mi.response_model_name.as_deref(),
            some(eq("claude-sonnet-4-20250514"))
        );
        expect_that!(mi.operation.as_deref(), some(eq("chat")));
        expect_that!(mi.input_tokens, some(eq(13)));
        expect_that!(mi.output_tokens, some(eq(8)));
        expect_that!(mi.finish_reason, some(eq(FinishReason::Stop)));
        expect_that!(mi.provider_cache_write_input_tokens, some(eq(0)));
    }

    // ── Anthropic Tool Use ───────────────────────────────────────────────

    #[gtest]
    fn anthropic_tool_use_parses_two_spans() {
        let results = parse_otlp_jsonl(ANTHROPIC_TOOL_USE).expect("should parse");
        expect_that!(results.len(), eq(2));
        assert_required_fields(&results);
    }

    #[gtest]
    fn anthropic_tool_call_span() {
        let results = parse_otlp_jsonl(ANTHROPIC_TOOL_USE).expect("should parse");
        let mi = &results[0];

        expect_that!(mi.model_name, eq("claude-sonnet-4-20250514"));
        expect_that!(
            mi.provider_response_id.as_deref(),
            some(eq("msg_01DuKSqmpQ4Y7vWfZthTqcZh"))
        );
        expect_that!(mi.input_tokens, some(eq(390)));
        expect_that!(mi.output_tokens, some(eq(67)));
        expect_that!(mi.finish_reason, some(eq(FinishReason::ToolCall)));
        expect_that!(
            mi.raw_response.as_deref(),
            some(contains_substring("get_weather"))
        );
    }

    #[gtest]
    fn anthropic_tool_result_followup_span() {
        let results = parse_otlp_jsonl(ANTHROPIC_TOOL_USE).expect("should parse");
        let mi = &results[1];

        expect_that!(
            mi.provider_response_id.as_deref(),
            some(eq("msg_01Pg51hhLM7qXeVPUk2qmRBq"))
        );
        expect_that!(mi.input_tokens, some(eq(479)));
        expect_that!(mi.output_tokens, some(eq(28)));
        expect_that!(mi.finish_reason, some(eq(FinishReason::Stop)));
    }
}
