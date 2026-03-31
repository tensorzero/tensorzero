---
name: t0-cache-status
description: Quick overview of prompt caching effectiveness across all functions by querying the TensorZero gateway for cache hit rates and token usage.
argument-hint: [--gateway URL] [--top N] [--api-key KEY]
allowed-tools: Bash(curl *), Bash(jq *), Bash(printf *), Bash(sort *), Bash(column *)
---

# Prompt Cache Status Dashboard

Quick dashboard showing prompt caching effectiveness across all functions and variants. All data comes from the gateway API.

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

### Step 2: Fetch recent inferences across all functions

Query `list_inferences` without a function filter to get recent inferences across the system:

```bash
curl -s -X POST $AUTH_HEADER "${GATEWAY_URL}/v1/inferences/list_inferences" \
  -H "Content-Type: application/json" \
  -d '{"limit": 200}'
```

Group the results by `function_name` + `variant_name` to discover active function/variant pairs.

### Step 3: For each function+variant, get model inference details

For a sample of inference IDs per variant (up to `--top` count), fetch model-level data:

```bash
curl -s $AUTH_HEADER "${GATEWAY_URL}/internal/model_inferences/${INFERENCE_ID}"
```

This returns `{ "model_inferences": [...] }` with:

- `model_name` — the model used
- `model_provider_name` — the provider
- `input_tokens` — total input tokens
- `output_tokens` — total output tokens
- `provider_cache_read_input_tokens` — tokens served from cache (null if not reported)
- `provider_cache_write_input_tokens` — tokens written to cache (null if not reported)

### Step 4: Compute cache metrics per variant

For each variant, compute:

- Mean cache hit rate: `sum(provider_cache_read_input_tokens) / sum(input_tokens)`
- Mean input tokens per request
- Model name and provider (from model inference data)

If cache token fields are all null, report "No cache data" for that variant.

### Step 5: Display dashboard

Output a formatted table:

```
## Prompt Cache Status

| Function | Variant | Model | Provider | Inferences | Avg Input Tokens | Cache Hit Rate |
|----------|---------|-------|----------|------------|-----------------|----------------|
| my_func  | v1      | claude-sonnet-4-5 | anthropic | 20 | 3,450 | 78% |
| my_func  | v2      | gpt-4o | openai | 20 | 2,100 | 45% |
| other    | default | claude-haiku-4-5 | anthropic | 15 | 890 | 0% |

### Summary
- Total function+variant pairs with data: N
- Variants with cache hit rate > 50%: X / Y
- Variants needing attention (hit rate < 50%): Z
```

Sort by cache hit rate ascending so the worst performers are at the top.

### Step 6: Flag issues

For any variant where:

- Cache hit rate is 0% and input tokens > 1000: flag as "No caching detected"
- Cache hit rate < 50% with Anthropic model: flag as "Consider adding cache breakpoints"
- No token data available: note as "No cache data"

Suggest running `/t0-cache-audit <function_name> <variant_name>` for variants that need attention.

## Notes

- This is a read-only diagnostic. It does not modify any configuration.
- Cache statistics depend on the provider including usage data in responses.
- Some older inferences may not have cache statistics.
- If no inferences exist, report that the system has no recent inference data.
