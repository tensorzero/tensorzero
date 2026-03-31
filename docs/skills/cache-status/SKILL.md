---
name: cache-status
description: Quick overview of prompt caching effectiveness across all functions by querying the TensorZero gateway for cache hit rates and token usage.
argument-hint: [--gateway URL] [--top N] [--api-key KEY]
allowed-tools: Bash(python3 *)
---

# Prompt Cache Status Dashboard

Run the status script and display its output to the user:

```bash
python3 "${SKILL_DIR}/cache_status.py" $ARGUMENTS
```

The script queries the TensorZero gateway for recent inferences across all functions, computes cache hit rates per variant, and outputs a formatted dashboard table.

Progress messages go to stderr; only the final dashboard goes to stdout.
