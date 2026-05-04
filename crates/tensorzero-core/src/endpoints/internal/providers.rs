//! `GET /internal/providers/available` — used by the UI's model picker to
//! show a curated dropdown of providers and common models, with a flag
//! per provider indicating whether the gateway has credentials for it.
//!
//! The credential check is intentionally crude — it just looks for the
//! presence of the well-known env var. The UI uses this only to sort and
//! visually mark "credentialed" providers so users land on the ones they
//! can actually call. A user can still pick a non-credentialed provider
//! (or type a fully custom model string) and the gateway will fail at
//! inference time as it always has.
//!
//! The model lists per provider are static and intentionally short. They
//! grow rarely enough that bumping them in a follow-up PR is cheaper than
//! pulling them dynamically from each provider's `/models` endpoint
//! (which would also burn rate limits and depend on outbound network).
//! "Custom..." in the UI lets users override with any model name.

use axum::Json;
use serde::Serialize;

use crate::error::Error;

/// One row in the `/internal/providers/available` response.
#[derive(Debug, Clone, Serialize)]
pub struct ProviderInfo {
    /// The shorthand id used in `<provider>::<model>` model strings,
    /// e.g. `"openai"`, `"anthropic"`, `"gcp_vertex_gemini"`.
    pub id: &'static str,
    /// Human-readable label for the picker.
    pub display_name: &'static str,
    /// Whether the gateway sees credentials for this provider in its env.
    /// Cheap heuristic — true if the provider's primary env var is set.
    pub credential_present: bool,
    /// A short, hand-curated list of models known to work with this
    /// provider. Not exhaustive; users can type anything via "Custom".
    pub common_models: &'static [&'static str],
}

/// Top-level response.
#[derive(Debug, Clone, Serialize)]
pub struct AvailableProvidersResponse {
    pub providers: Vec<ProviderInfo>,
}

/// Static catalog of provider metadata. Order is the order returned to the
/// UI; the UI then re-sorts credentialed providers to the top. When adding
/// a provider here, also add its shorthand prefix to
/// `SHORTHAND_MODEL_PREFIXES` in `model.rs` if not already there.
const PROVIDER_CATALOG: &[(&str, &str, &str, &[&str])] = &[
    // (id, display_name, env var to probe, common_models)
    //
    // Each list mixes a fast/cheap tier and a frontier tier so users
    // landing on the picker for the first time see something they're
    // likely to actually pick. Provider-specific IDs only — anything
    // exotic should go through the "Custom..." path in the UI.
    //
    // **Verification policy:** the OpenAI, Anthropic, and Google AI
    // Studio entries below are verified against each provider's
    // `/v1/models` listing as of 2026-04-29. The other providers' lists
    // are best-effort and should be re-verified before they're shown to
    // users — see `feedback_verify_model_ids` in memory.
    (
        "openai",
        "OpenAI",
        "OPENAI_API_KEY",
        &[
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-5.4-mini",
            "gpt-5.5",
            "o4-mini",
            "o3",
        ],
    ),
    (
        "anthropic",
        "Anthropic",
        "ANTHROPIC_API_KEY",
        // No `*-latest` aliases — Anthropic retired them. Pin to the
        // versioned IDs returned by the live `/v1/models` listing.
        &[
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-opus-4-7",
        ],
    ),
    (
        "google_ai_studio_gemini",
        "Google AI Studio (Gemini)",
        "GOOGLE_AI_STUDIO_API_KEY",
        &[
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-3-pro-preview",
            "gemini-3.1-pro-preview",
        ],
    ),
    (
        "gcp_vertex_gemini",
        "Google Vertex (Gemini)",
        // Either ADC path is acceptable; we report `true` if any is set.
        // The probe function below handles the fallback.
        "GOOGLE_APPLICATION_CREDENTIALS",
        &[
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-3-pro-preview",
        ],
    ),
    (
        "gcp_vertex_anthropic",
        "Google Vertex (Anthropic)",
        "GOOGLE_APPLICATION_CREDENTIALS",
        // Vertex pins Claude to `@YYYYMMDD` snapshots that lag Anthropic's
        // direct API by days/weeks. Verify in Vertex Model Garden when
        // bumping — date suffixes are NOT interchangeable across the two
        // surfaces.
        &[
            "claude-haiku-4-5@20251001",
            "claude-sonnet-4-5@20250929",
            "claude-opus-4-5@20251101",
        ],
    ),
    (
        "mistral",
        "Mistral",
        "MISTRAL_API_KEY",
        &[
            "mistral-small-latest",
            "mistral-medium-latest",
            "mistral-large-latest",
            "codestral-latest",
        ],
    ),
    (
        "fireworks",
        "Fireworks",
        "FIREWORKS_API_KEY",
        &[
            "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "accounts/fireworks/models/llama4-scout-instruct-basic",
            "accounts/fireworks/models/llama4-maverick-instruct-basic",
            "accounts/fireworks/models/deepseek-v3",
            "accounts/fireworks/models/deepseek-r1",
        ],
    ),
    (
        "together",
        "Together",
        "TOGETHER_API_KEY",
        &[
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1",
        ],
    ),
    (
        "groq",
        "Groq",
        "GROQ_API_KEY",
        &[
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "deepseek-r1-distill-llama-70b",
        ],
    ),
    (
        "xai",
        "xAI",
        "XAI_API_KEY",
        &["grok-3-mini", "grok-3", "grok-4"],
    ),
    (
        "deepseek",
        "DeepSeek",
        "DEEPSEEK_API_KEY",
        // DeepSeek keeps these two aliases stable; the underlying
        // weights are upgraded silently.
        &["deepseek-chat", "deepseek-reasoner"],
    ),
    (
        "openrouter",
        "OpenRouter",
        "OPENROUTER_API_KEY",
        &[
            "openai/gpt-4.1-mini",
            "openai/gpt-4.1",
            "anthropic/claude-sonnet-4.5",
            "anthropic/claude-opus-4.5",
            "google/gemini-2.5-pro",
            "deepseek/deepseek-r1",
        ],
    ),
    (
        "hyperbolic",
        "Hyperbolic",
        "HYPERBOLIC_API_KEY",
        &[
            "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1",
        ],
    ),
    (
        "dummy",
        "Dummy (testing)",
        "TENSORZERO_DUMMY_PROVIDER_ENABLED",
        &["good", "echo", "json"],
    ),
];

/// Multi-env-var fallback for providers whose credentials can come from
/// several places (notably GCP, where any of three env vars is enough).
/// Returns true if any of the listed env vars is set to a non-empty value.
fn any_env_set(vars: &[&str]) -> bool {
    vars.iter()
        .any(|name| std::env::var(name).map(|v| !v.is_empty()).unwrap_or(false))
}

/// Provider-specific credential probe. Most providers are a single env
/// var; GCP/AWS allow several alternatives. Kept here rather than baked
/// into the catalog so the catalog stays a plain data table.
fn credential_present(provider_id: &str, primary_env_var: &str) -> bool {
    match provider_id {
        // GCP credentials can come from any of these.
        "gcp_vertex_gemini" | "gcp_vertex_anthropic" => any_env_set(&[
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GCP_VERTEX_CREDENTIALS_PATH",
            "GCP_VERTEX_CREDENTIALS_JSON",
        ]),
        // Default: just probe the primary env var.
        _ => any_env_set(&[primary_env_var]),
    }
}

/// Handler for `GET /internal/providers/available`.
#[expect(
    clippy::unused_async,
    reason = "axum handlers must be async; the body is synchronous because the catalog is in-memory"
)]
pub async fn list_available_providers_handler() -> Result<Json<AvailableProvidersResponse>, Error> {
    let providers = PROVIDER_CATALOG
        .iter()
        .map(|(id, display_name, env_var, common_models)| ProviderInfo {
            id,
            display_name,
            credential_present: credential_present(id, env_var),
            common_models,
        })
        .collect();

    Ok(Json(AvailableProvidersResponse { providers }))
}
