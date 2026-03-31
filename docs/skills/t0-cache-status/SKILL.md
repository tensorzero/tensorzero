---
name: t0-cache-status
description: Quick overview of prompt caching effectiveness across all functions by querying the TensorZero gateway for cache hit rates and token usage.
disable-model-invocation: true
argument-hint: [--gateway URL] [--top N] [--api-key KEY]
allowed-tools: Bash(curl *), Bash(jq *), Bash(printf *), Bash(sort *), Bash(column *)
---

# Prompt Cache Status Dashboard

Quick dashboard showing prompt caching effectiveness across all functions and variants.

## Arguments

- `--gateway` — Gateway URL (default: `http://localhost:3000`)
- `--top` — Number of inferences per variant to sample (default: 20)
- `--api-key` — TensorZero API key for authenticated gateways (optional)

## Authentication

TensorZero gateway auth is **optional** — controlled by `[gateway.auth] enabled` in `tensorzero.toml`.

- When auth is **disabled** (default for local dev): no key needed.
- When auth is **enabled**: pass `--api-key <key>` or set `TENSORZERO_API_KEY` env var.
- The key is sent as `Authorization: Bearer <key>` on every request.

If the first request returns 401, tell the user they need to provide an API key.

## Procedure

### Step 1: Parse arguments and resolve auth

Parse `$ARGUMENTS` for `--gateway URL`, `--top N`, and `--api-key KEY` flags.
Default gateway is `http://localhost:3000`, default top is `20`.

Resolve the API key from (in order):

1. `--api-key` argument
2. `TENSORZERO_API_KEY` environment variable
3. None (unauthenticated)

Build a shared `AUTH_HEADER`: if a key is present, use `-H "Authorization: Bearer $KEY"` on all curl calls.

### Step 2: Fetch live config

```bash
curl -s $AUTH_HEADER "${GATEWAY_URL}/internal/config"
```

Extract all function+variant pairs (skip `tensorzero::` built-in functions).

### Step 3: For each function+variant, sample recent inferences

For each pair, fetch recent inferences:

```bash
curl -s -X POST $AUTH_HEADER "${GATEWAY_URL}/v1/inferences/list_inferences" \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "FUNC",
    "variant_name": "VAR",
    "limit": TOP
  }'
```

### Step 4: Extract cache metrics from provider responses

For each inference, look at the raw model output for cache statistics:

**Anthropic** (claude-\* models):

- `usage.input_tokens` — total input tokens
- `usage.cache_read_input_tokens` — tokens served from cache
- `usage.cache_creation_input_tokens` — tokens written to cache

**OpenAI** (gpt-\* models):

- `usage.prompt_tokens` — total prompt tokens
- `usage.prompt_tokens_details.cached_tokens` — cached tokens

Compute per-variant:

- Mean cache hit rate: `cached_tokens / total_input_tokens`
- Mean input tokens per request
- Whether cache breakpoints are configured (Anthropic only)

### Step 5: Display dashboard

Output a formatted table:

```
## Prompt Cache Status

| Function | Variant | Model | Inferences | Avg Input Tokens | Cache Hit Rate | Breakpoints |
|----------|---------|-------|------------|-----------------|----------------|-------------|
| my_func  | v1      | claude-sonnet-4-5 | 20 | 3,450 | 78% | Yes |
| my_func  | v2      | gpt-4o | 20 | 2,100 | 45% | N/A (auto) |
| other    | default | claude-haiku-4-5 | 15 | 890 | 0% | No |

### Summary
- Total functions: N
- Variants with caching active: X / Y
- Variants needing attention (hit rate < 50%): Z
```

Sort by cache hit rate ascending so the worst performers are at the top.

### Step 6: Flag issues

For any variant where:

- Cache hit rate is 0% and input tokens > 1000: flag as "No caching detected"
- Cache hit rate < 50% with Anthropic model and no breakpoints: flag as "Missing cache breakpoints"
- No usage data available: note as "No usage data"

## Notes

- This is a read-only diagnostic. It does not modify any configuration.
- Cache statistics depend on the provider including usage data in responses.
- Some older inferences may not have cache statistics.
