---
name: prompt-audit
description: Run a prompt caching audit on a TensorZero function+variant by querying the gateway for recent inferences. Detects tool ordering issues, missing cache breakpoints, and computes cache hit rates.
argument-hint: <function_name> [variant_name] [--gateway URL] [--limit N] [--api-key KEY]
allowed-tools: Bash(python3 *)
---

# Prompt Caching Audit

Run the audit script and display its output to the user:

```bash
python3 "${SKILL_DIR}/prompt_audit.py" $ARGUMENTS
```

The script queries the TensorZero gateway API for recent inferences and checks:

1. **Tool ordering stability** — non-deterministic ordering breaks prefix caching
2. **Cache breakpoints** — Anthropic models need explicit `cache_control` via `extra_body`
3. **Cache hit rate** — computed from `provider_cache_read_input_tokens`

Progress messages go to stderr; only the final report goes to stdout.
