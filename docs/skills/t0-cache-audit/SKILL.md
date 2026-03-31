---
name: t0-cache-audit
description: Run a prompt caching audit on a TensorZero function+variant by querying the gateway for recent inferences. Detects tool ordering issues, missing cache breakpoints, and estimates savings.
argument-hint: <function_name> [variant_name] [--gateway URL] [--limit N]
allowed-tools: Bash(curl *), Bash(jq *)
---

# Prompt Caching Audit

Run a prompt caching analysis for a TensorZero function+variant pair. All data comes from the gateway API — no config files needed.

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

If the first request returns 401, tell the user they need to provide an API key.

## Procedure

### Step 1: Parse arguments and resolve auth

Parse `$ARGUMENTS` for the function name, optional variant name, and flags.
Default gateway URL is `http://localhost:3000`. Default limit is `100`.

Resolve the API key from (in order):

1. `--api-key` argument
2. `TENSORZERO_API_KEY` environment variable
3. None (unauthenticated)

Build a shared `AUTH_HEADER` variable: if a key is present, use `-H "Authorization: Bearer $KEY"` on all curl calls; otherwise omit it.

### Step 2: Fetch recent inferences

```bash
curl -s -X POST $AUTH_HEADER "${GATEWAY_URL}/v1/inferences/list_inferences" \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "FUNCTION_NAME",
    "variant_name": "VARIANT_NAME",
    "limit": LIMIT
  }'
```

This returns `{ "inferences": [...] }` with each inference containing `inference_id`, `variant_name`, `input`, `output`, `provider_tools`, `extra_body`, etc.

If no variant was specified, omit `variant_name` from the request body — the response will include inferences across all variants. Group by `variant_name` to audit each one.

### Step 3: Fetch model inference details for cache stats

For each inference (or a representative sample of ~10-20), get the model-level data:

```bash
curl -s $AUTH_HEADER "${GATEWAY_URL}/internal/model_inferences/${INFERENCE_ID}"
```

This returns `{ "model_inferences": [...] }` with fields:

- `model_name` — the model used (e.g. `claude-sonnet-4-5`, `gpt-4o`, `dummy::good`)
- `model_provider_name` — the provider used (e.g. `anthropic`, `openai`, `dummy`)
- `input_tokens` — total input tokens
- `output_tokens` — total output tokens
- `provider_cache_read_input_tokens` — tokens served from provider cache (null if not reported)
- `provider_cache_write_input_tokens` — tokens written to provider cache (null if not reported)
- `cached` — whether the inference was cached
- `system` — the system prompt text
- `input_messages` — the input messages sent to the model

### Step 4: Analyze for prompt caching issues

For each variant's inferences, perform these checks:

#### Check 1: Tool Ordering Stability

Extract tool definitions from each inference's `provider_tools` field (from Step 2).
Serialize each inference's tools to a canonical JSON string.
Count how many distinct tool orderings appear.

**Verdict:**

- If tools appear in >1 order across inferences: **FAIL** — "Tool definitions appear in N different orders across M inferences. Non-deterministic tool ordering breaks prefix caching."
- If tools are stable or no tools: **PASS** or **INFO**

#### Check 2: Cache Breakpoints (Anthropic models)

Determine the provider from the model inference data (Step 3). For Anthropic models (model_provider_name contains `anthropic` or model_name starts with `claude`):

- Check if the inference's `extra_body` (from Step 2) contains any `cache_control` pointers
- If the system prompt (from Step 3) is long (>1024 tokens estimated at ~4 chars/token) and stable across inferences, but no cache breakpoints are configured: **FAIL**
- If cache breakpoints are configured: **PASS**
- For non-Anthropic models (OpenAI, Google, etc.): **INFO** — "This provider uses automatic prefix caching"

#### Check 3: Cache Hit Rate (from model inference data)

Using the model inference data from Step 3, compute:

- Cache hit rate: `sum(provider_cache_read_input_tokens) / sum(input_tokens)`
- Cache write rate: `sum(provider_cache_write_input_tokens) / sum(input_tokens)`

If all cache token fields are null, report "No cache data available — provider does not report cache statistics."

Flag if cache hit rate < 50% and there are significant input tokens.

### Step 5: Report results

Output a clear report with:

```
## Cache Audit: function_name / variant_name
Model: <model_name> (provider: <model_provider_name>)
Sample size: <N> inferences

### Tool Ordering: PASS|FAIL|INFO
<details>

### Cache Breakpoints: PASS|FAIL|INFO|N/A
<details>

### Cache Hit Rate: <X%> | No data
<details>

### Recommendations
1. <actionable recommendation>
2. ...
```

## Important Notes

- The gateway must be running and accessible at the specified URL.
- For Anthropic models, cache breakpoints are set via `extra_body` in the variant config.
- Tool ordering issues affect ALL providers (prefix and content-block caching).
- System prompt stability is critical for cache effectiveness.
- If no inferences exist for the function/variant, report that and suggest running some inferences first.
