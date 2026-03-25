@AGENTS.md


# Current Task
You are working on issue #7034: **Add cache token parsing for all providers**

GitHub: https://github.com/tensorzero/tensorzero/pull/7034

Labels: has-merge-conflicts

## Issue Description

## Summary
- Adds provider-specific cache token extraction to all providers that support prompt caching:
  - **OpenAI/Groq/XAI/OpenRouter/Together**: `prompt_tokens_details.cached_tokens`
  - **Anthropic/GCP Vertex Anthropic**: `cache_read_input_tokens` / `cache_creation_input_tokens`
  - **AWS Bedrock**: `cacheReadInputTokenCount` / `cacheWriteInputTokenCount`
  - **DeepSeek**: `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` (introduces `DeepSeekUsage` type)
  - **Fireworks**: `x-fireworks-cached-prompt-tokens` HTTP header
  - **Mistral**: `prompt_tokens_details.cached_tokens`
  - **GCP Vertex Gemini / Google AI Studio**: `cachedContentTokenCount`
  - **vLLM**: `num_cached_tokens` from `prompt_tokens_details`
- Adds `cache.rs` documentation registry in `tensorzero-types-providers` mapping each provider's cache API to TensorZero's normalized fields
- Adds new provider-specific serde types (`DeepSeekUsage`, `MistralPromptTokensDetails`, `OpenAIPromptTokensDetails`, etc.)

Stacked on #7033.

## Test plan
- [x] `cargo check --all-targets --all-features` passes
- [x] `cargo clippy --all-targets --all-features -- -D warnings` passes
- [ ] Provider-specific unit tests included in provider files
- [ ] E2e cache token tests (in later PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
## Instructions
- Work on the branch `feat/cache-tokens-3-provider-types`
- Focus on resolving this issue
- Create atomic commits with clear messages
- When done, let the user know so they can create a PR
