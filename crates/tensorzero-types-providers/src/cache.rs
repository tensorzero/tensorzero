//! Cache token field mappings for all supported providers.
//!
//! This module is the single source of truth for how each provider reports
//! prompt-caching metrics in their API responses. Every provider that supports
//! caching should be listed here with its wire-format fields and the mapping
//! to TensorZero's internal `Usage` struct.
//!
//! # Internal `Usage` fields
//!
//! | Field | Meaning |
//! |-------|---------|
//! | `provider_cache_read_input_tokens` | Tokens served from cache (cheaper) |
//! | `provider_cache_write_input_tokens` | Tokens written to cache (may cost more) |
//!
//! `None` = provider doesn't report this metric.
//! `Some(0)` = provider supports caching but zero tokens were cached.
//!
//! # Provider cache field mappings
//!
//! ## OpenAI-compatible providers
//!
//! These providers return `prompt_tokens_details.cached_tokens` and reuse
//! [`OpenAIPromptTokensDetails`](super::openai::OpenAIPromptTokensDetails).
//!
//! | Provider | `cache_read` source | `cache_write` source | Mechanism |
//! |----------|---------------------|----------------------|-----------|
//! | **OpenAI** | `prompt_tokens_details.cached_tokens` | — | Automatic (>= 1024 tokens) |
//! | **Groq** | `prompt_tokens_details.cached_tokens` | — | Automatic |
//! | **xAI** | `prompt_tokens_details.cached_tokens` | — | Automatic |
//! | **OpenRouter** | `prompt_tokens_details.cached_tokens` | — | Varies by underlying provider |
//!
//! ## Anthropic-format providers
//!
//! | Provider | `cache_read` source | `cache_write` source | Mechanism |
//! |----------|---------------------|----------------------|-----------|
//! | **Anthropic** | `cache_read_input_tokens` | `cache_creation_input_tokens` | Explicit (`cache_control`) |
//! | **GCP Vertex Anthropic** | `cache_read_input_tokens` | `cache_creation_input_tokens` | Explicit (`cache_control`) |
//! | **AWS Bedrock** | `cacheReadInputTokenCount` | `cacheWriteInputTokenCount` | Explicit (`cachePoint`) |
//!
//! ## Google Gemini providers
//!
//! | Provider | `cache_read` source | `cache_write` source | Mechanism |
//! |----------|---------------------|----------------------|-----------|
//! | **GCP Vertex Gemini** | `usageMetadata.cachedContentTokenCount` | — | Implicit (2.5+) / Explicit (CachedContent API) |
//! | **Google AI Studio Gemini** | `usageMetadata.cachedContentTokenCount` | — | Implicit (2.5+) / Explicit (CachedContent API) |
//!
//! Note: GCP Vertex Gemini (`aiplatform.googleapis.com`) returns `cachedContentTokenCount`
//! on cache hits, but only opportunistically — it's not guaranteed even with identical
//! prompts. Google AI Studio (`generativelanguage.googleapis.com`) does NOT return this
//! field at all as of March 2026. Both providers parse the field correctly if present.
//!
//! ## DeepSeek (unique format)
//!
//! | Provider | `cache_read` source | `cache_write` source | Mechanism |
//! |----------|---------------------|----------------------|-----------|
//! | **DeepSeek** | `prompt_cache_hit_tokens` | `prompt_cache_miss_tokens` | Automatic |
//!
//! DeepSeek uses top-level usage fields instead of `prompt_tokens_details`.
//! Parsed via [`DeepSeekUsage`](super::deepseek::DeepSeekUsage).
//!
//! ## Fireworks (HTTP headers)
//!
//! | Provider | `cache_read` source | `cache_write` source | Mechanism |
//! |----------|---------------------|----------------------|-----------|
//! | **Fireworks** | HTTP header `fireworks-cached-prompt-tokens` | — | Automatic |
//!
//! Fireworks returns cache info in HTTP response headers, not in the JSON body.
//!
//! ## Mistral
//!
//! | Provider | `cache_read` source | `cache_write` source | Mechanism |
//! |----------|---------------------|----------------------|-----------|
//! | **Mistral** | `prompt_tokens_details.cached_tokens` | — | Automatic |
//!
//! ## Providers with caching NOT yet exposed in JSON responses
//!
//! | Provider | Notes |
//! |----------|-------|
//! | **Together** | Transparent backend caching, no token counts in response |
//! | **Hyperbolic** | No prompt caching support |
//! | **vLLM** | Parsed via `prompt_tokens_details.cached_tokens` (OpenAI-compatible), but not all deployments report it |
//! | **SGLang** | Parsed via `prompt_tokens_details.cached_tokens` (OpenAI-compatible), but not all deployments report it |
//!
//! # Adding a new provider
//!
//! 1. Add a row to the appropriate table above.
//! 2. If the provider uses `prompt_tokens_details.cached_tokens`, reuse
//!    [`OpenAIPromptTokensDetails`](super::openai::OpenAIPromptTokensDetails)
//!    in the provider's usage struct.
//! 3. In the provider's `From<ProviderUsage> for Usage` impl, map the fields.
//! 4. Add the provider to `cache_input_tokens_inference` in the e2e test setup.
//! 5. Run the cache token e2e tests to verify.

// This module is documentation-only. All cache-related types live in their
// respective provider modules (e.g. `openai::OpenAIPromptTokensDetails`).
// The canonical list of which providers support caching is the doc comment above.
