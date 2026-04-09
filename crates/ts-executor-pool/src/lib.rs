pub mod error;
pub mod inference;
pub mod llm_query;
pub mod runtime;
pub mod state;
pub mod tensorzero_client;
pub mod tool_result;
pub mod ts_checker;
pub use bip39_uuid_substitution as uuid_substitution;

pub use error::TsError;

// These will eventually become private as we move more code into this crate.

/// SES (Secure ECMAScript) library source, downloaded at build time.
/// Defines `lockdown`, `harden`, and `Compartment` on `globalThis`.
pub static SES_JS: &str = include_str!(env!("SES_JS_PATH"));

/// SES initialization script: calls `lockdown()` and defines the
/// compartment factory used by `create_rlm_runtime`.
pub static SES_INIT_JS: &str = include_str!("./js/ses_init.js");

/// Truncate `text` to at most `max_chars` Unicode characters, returning a `&str`
/// slice. If the text is already within the limit it is returned unchanged.
pub fn truncate_to_chars(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}

/// Context for inference requests, containing tenant identification.
///
/// This is passed to inference functions to ensure all calls are tagged
/// with the correct organization and workspace.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceContext {
    pub session_id: uuid::Uuid,
    pub event_id: uuid::Uuid,
    pub organization_id: String,
    pub workspace_id: String,
}

/// Extra tags to attach to every inference request.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ExtraInferenceTags(pub std::collections::BTreeMap<String, String>);

impl ExtraInferenceTags {
    pub fn new(ctx: &InferenceContext) -> Self {
        Self(std::collections::BTreeMap::from([
            ("organization_id".to_string(), ctx.organization_id.clone()),
            ("workspace_id".to_string(), ctx.workspace_id.clone()),
            ("session_id".to_string(), ctx.session_id.to_string()),
            ("event_id".to_string(), ctx.event_id.to_string()),
        ]))
    }
}

/// Configuration for the RLM context management strategy.
#[derive(Debug, Clone)]
pub struct RlmConfig {
    /// Maximum number of LLM ↔ REPL iterations before giving up.
    pub max_iterations: usize,
    /// Maximum recursion depth for nested `llm_query` calls.
    pub max_depth: u32,
    /// Character limit below which output is considered manageable.
    pub char_limit: usize,
    /// Maximum wall-clock seconds allowed for a single JS code block execution.
    ///
    /// If a code block runs longer than this (e.g. synchronous infinite loop),
    /// the V8 isolate is terminated and execution returns a timeout error.
    pub execution_timeout_secs: u64,
    /// Optional TypeScript type declarations for the tool output.
    ///
    /// When present, included in the initial RLM prompt so the LLM knows the
    /// exact shape of `globalThis.context`. Should be self-contained (no imports).
    pub output_ts_type: Option<String>,
}

impl RlmConfig {
    /// Sensible operational defaults.
    ///
    /// Instructions are now supplied per-call via `ctx.instructions` in the
    /// tool's augmentation parameters.
    pub fn new() -> Self {
        Self {
            max_iterations: 10,
            max_depth: 3,
            char_limit: 10_000,
            execution_timeout_secs: 30,
            output_ts_type: None,
        }
    }
}

impl Default for RlmConfig {
    fn default() -> Self {
        Self::new()
    }
}
