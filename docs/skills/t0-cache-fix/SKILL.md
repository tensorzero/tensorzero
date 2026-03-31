---
name: t0-cache-fix
description: Generate and optionally apply prompt caching fixes for a TensorZero variant — adds cache breakpoints via extra_body config or fixes tool ordering.
disable-model-invocation: true
argument-hint: <function_name> <variant_name> [--gateway URL] [--apply] [--api-key KEY]
allowed-tools: Bash(curl *), Bash(jq *), Read, Grep, Glob, Write, Edit
---

# Prompt Cache Fix

Generate and optionally apply prompt caching configuration fixes for a specific TensorZero function+variant.

## Arguments

- `$0` — function_name (required)
- `$1` — variant_name (required)
- `--gateway` — Gateway URL (default: `http://localhost:3000`)
- `--apply` — Actually write the config changes (default: dry-run)
- `--dry-run` — Show what would change without applying (default behavior)
- `--api-key` — TensorZero API key for authenticated gateways (optional)

## Authentication

TensorZero gateway auth is **optional** — controlled by `[gateway.auth] enabled` in `tensorzero.toml`.

- When auth is **disabled** (default for local dev): no key needed.
- When auth is **enabled**: pass `--api-key <key>` or set `TENSORZERO_API_KEY` env var.
- The key is sent as `Authorization: Bearer <key>` on every request.

If the first request returns 401, tell the user they need to provide an API key.

## Procedure

### Step 1: Parse arguments, resolve auth, and fetch current config

Resolve the API key from (in order):

1. `--api-key` argument
2. `TENSORZERO_API_KEY` environment variable
3. None (unauthenticated)

Build a shared `AUTH_HEADER`: if a key is present, use `-H "Authorization: Bearer $KEY"` on all curl calls.

```bash
curl -s $AUTH_HEADER "${GATEWAY_URL}/internal/config"
```

Extract the variant config for the specified function+variant. Identify:

- Model name and provider (anthropic, openai, google, etc.)
- Current `extra_body` configuration (if any)
- Tool definitions bound to the function
- System prompt template

### Step 2: Fetch recent inferences for analysis

```bash
curl -s -X POST $AUTH_HEADER "${GATEWAY_URL}/v1/inferences/list_inferences" \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "FUNC",
    "variant_name": "VAR",
    "limit": 50
  }'
```

### Step 3: Diagnose issues

Analyze the inferences to identify specific problems:

#### Issue A: Missing Cache Breakpoints (Anthropic only)

For Anthropic models, check if `extra_body` has `cache_control` entries.
If not, determine optimal breakpoint placement:

1. **System prompt breakpoint**: If system is stable across inferences and long enough (>1024 tokens ~= 4096 chars), recommend a breakpoint after system.
   - Pointer: `/system/0/cache_control`
   - Value: `{"type": "ephemeral"}`

2. **Tools breakpoint**: If tools are stable and present, recommend a breakpoint after the last tool.
   - Pointer: `/tools/-1/cache_control`
   - Value: `{"type": "ephemeral"}`

3. **Static message breakpoints**: If early messages (e.g., few-shot examples) are identical across inferences, recommend breakpoints after them.
   - Pointer: `/messages/N/content/0/cache_control`
   - Value: `{"type": "ephemeral"}`

Maximum 4 breakpoints (Anthropic API limit). Place them at the boundary of the longest stable prefix.

#### Issue B: Tool Ordering

Check if tools appear in different orders across inferences. If so, recommend:

- Setting `tool_choice` to a deterministic option
- Or sorting tools alphabetically in the variant config

### Step 4: Generate fix

For breakpoint fixes, generate the `extra_body` config to add to the variant:

```toml
# Add to your variant configuration
[functions.FUNCTION.variants.VARIANT]
# ... existing config ...

[functions.FUNCTION.variants.VARIANT.extra_body]
data = [
    { pointer = "/system/0/cache_control", value = { type = "ephemeral" } },
    { pointer = "/tools/-1/cache_control", value = { type = "ephemeral" } },
]
```

Also generate the equivalent JSON for the config write API:

```json
{
  "extra_body": {
    "data": [
      {
        "pointer": "/system/0/cache_control",
        "value": { "type": "ephemeral" }
      },
      { "pointer": "/tools/-1/cache_control", "value": { "type": "ephemeral" } }
    ]
  }
}
```

### Step 5: Show or apply

**Dry-run (default):** Display the recommended changes with explanation of expected impact.

Estimate savings:

- Anthropic cached read tokens cost ~90% less than regular input tokens
- Calculate: `cacheable_tokens * 0.9 * cost_per_token * requests_per_day`

**Apply (--apply flag):** If the user passed `--apply`:

1. Find the `tensorzero.toml` config file in the project
2. Add or update the `extra_body` section for the variant
3. Show the diff of changes made

### Step 6: Verify (if applied)

If changes were applied, suggest:

```
After restarting the gateway with the new config, run:
  /cache-audit FUNCTION VARIANT
to verify the fix is working.
```

## Important Notes

- Anthropic allows max 4 cache breakpoints per request.
- Breakpoints only help if the content before them is stable across requests.
- For OpenAI/Google models, caching is automatic — no breakpoints needed. Focus on tool ordering only.
- Cache breakpoints use `{"type": "ephemeral"}` which caches for ~5 minutes (Anthropic default).
- The `extra_body` config in TensorZero supports variant-level, model-provider-level, and always-level overrides.
