---
name: t0-cache-audit
description: Run a prompt caching audit on a TensorZero function+variant by querying the gateway API directly. Detects tool ordering issues, missing cache breakpoints, and estimates savings.
disable-model-invocation: true
argument-hint: <function_name> [variant_name] [--gateway URL] [--limit N]
allowed-tools: Bash(curl *), Bash(jq *), Read, Grep, Glob
---

# Prompt Caching Audit via Gateway API

Run a prompt caching analysis for a TensorZero function+variant pair by querying the gateway API directly.

## Arguments

- `$0` — function_name (required)
- `$1` — variant_name (optional, audits all variants if omitted)
- `--gateway` — Gateway URL (default: `http://localhost:3000`)
- `--limit` — Number of recent inferences to analyze (default: 100)
- `--api-key` — TensorZero API key for authenticated gateways (optional)

## Authentication

TensorZero gateway auth is **optional** — controlled by `[gateway.auth] enabled` in `tensorzero.toml`.

- When auth is **disabled** (default for local dev): no key needed.
- When auth is **enabled**: pass `--api-key <key>` or set `TENSORZERO_API_KEY` env var.
- The key is sent as `Authorization: Bearer <key>` on every request.
- API keys are created via `tensorzero --create-api-key` and are scoped to an org+workspace.

If the first request returns 401, tell the user they need to provide an API key.

## Procedure

### Step 1: Resolve gateway URL, auth, and parse arguments

Parse `$ARGUMENTS` for the function name, optional variant name, and flags.
Default gateway URL is `http://localhost:3000`. Default limit is `100`.

Resolve the API key from (in order):

1. `--api-key` argument
2. `TENSORZERO_API_KEY` environment variable
3. None (unauthenticated)

Build a shared `AUTH_HEADER` variable: if a key is present, use `-H "Authorization: Bearer $KEY"` on all curl calls; otherwise omit it.

### Step 2: Get the config to discover variants

```bash
curl -s $AUTH_HEADER "${GATEWAY_URL}/internal/config" | jq '.functions["FUNCTION_NAME"].variants | keys[]'
```

If a specific variant was given, use only that one. Otherwise, audit all variants.

### Step 3: For each variant, fetch recent inferences

```bash
curl -s -X POST $AUTH_HEADER "${GATEWAY_URL}/v1/inferences/list_inferences" \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "FUNCTION_NAME",
    "variant_name": "VARIANT_NAME",
    "limit": LIMIT,
    "order_by": [{"term": "timestamp", "direction": "desc"}]
  }' | jq '.inferences'
```

### Step 4: Analyze for prompt caching issues

For each variant's inferences, perform these checks:

#### Check 1: Tool Ordering Stability

Extract the tool definitions from each inference's `input.messages` or the function config.
Serialize each inference's tools to a canonical JSON string.
Count how many distinct tool orderings appear.

**Verdict:**

- If tools appear in >1 order across inferences: **FAIL** — "Tool definitions appear in N different orders across M inferences. Non-deterministic tool ordering breaks prefix caching."
- If tools are stable or no tools: **PASS** or **INFO**

#### Check 2: Cache Breakpoints (Anthropic models)

Check the model name from the variant config. For Anthropic models (claude-\*):

- Check if `extra_body` contains any `cache_control` pointers
- Check the `extra_body` field in the inference data for `cache_control` entries
- If the system prompt is long (>1024 tokens estimated at ~4 chars/token) and stable across inferences, but no cache breakpoints are configured: **FAIL**
- If cache breakpoints are configured: **PASS**
- For non-Anthropic models (OpenAI, Google): **INFO** — "This provider uses automatic prefix caching"

#### Check 3: Cache Hit Rate (if usage data available)

Look in each inference's raw provider response for cache statistics:

- Anthropic: `usage.cache_read_input_tokens` / `usage.input_tokens`
- OpenAI: `usage.prompt_tokens_details.cached_tokens` / `usage.prompt_tokens`

If available, compute cache hit rate and flag if < 50%.

### Step 5: Report results

Output a clear report with:

```
## Cache Audit: function_name / variant_name
Model: <model_name>
Sample size: <N> inferences

### Tool Ordering: PASS|FAIL|INFO
<details>

### Cache Breakpoints: PASS|FAIL|INFO
<details>

### Cache Hit Rate: <X%> (if data available)
<details>

### Recommendations
1. <actionable recommendation>
2. ...
```

## Important Notes

- This skill queries the gateway API directly. It does NOT use any autopilot-specific code.
- The gateway must be running and accessible at the specified URL.
- For Anthropic models, cache breakpoints are set via `extra_body` in the variant config.
- Tool ordering issues affect ALL providers (prefix and content-block caching).
- System prompt stability is critical for cache effectiveness.
