#!/usr/bin/env python3
"""Prompt cache status dashboard for TensorZero.

Queries the gateway API for recent inferences across all functions,
computes cache hit rates per variant, and displays a summary table.
"""

import json
import sys
import urllib.error
import urllib.request
from collections import defaultdict


def api_post(url, data, headers=None):
    headers = headers or {}
    headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print(
                "ERROR: Gateway returned 401 Unauthorized. Pass --api-key <key> or set TENSORZERO_API_KEY.",
                file=sys.stderr,
            )
            sys.exit(1)
        body = e.read().decode()
        print(f"ERROR: Gateway returned {e.code}: {body}", file=sys.stderr)
        sys.exit(1)


def api_get(url, headers=None):
    headers = headers or {}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"ERROR: Gateway returned {e.code}: {body}", file=sys.stderr)
        sys.exit(1)


def parse_args(argv):
    gateway = "http://localhost:3000"
    top = 20
    api_key = None

    import os

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--gateway" and i + 1 < len(argv):
            gateway = argv[i + 1]
            i += 2
        elif arg == "--top" and i + 1 < len(argv):
            top = int(argv[i + 1])
            i += 2
        elif arg == "--api-key" and i + 1 < len(argv):
            api_key = argv[i + 1]
            i += 2
        else:
            i += 1

    if not api_key:
        api_key = os.environ.get("TENSORZERO_API_KEY")

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return gateway, top, headers


def main():
    gateway, top, headers = parse_args(sys.argv[1:])

    print("Fetching recent inferences...", file=sys.stderr)
    data = api_post(f"{gateway}/v1/inferences/list_inferences", {"limit": 200}, headers)
    inferences = data.get("inferences", [])

    if not inferences:
        print("\nNo recent inferences found. The system has no inference data to analyze.")
        sys.exit(0)

    # Group by function + variant
    groups = defaultdict(list)
    for inf in inferences:
        key = (inf["function_name"], inf["variant_name"])
        groups[key].append(inf)

    print(f"Found {len(inferences)} inferences across {len(groups)} function+variant pair(s).", file=sys.stderr)

    # Collect stats per variant
    rows = []
    for (func, variant), infs in sorted(groups.items()):
        sample = infs[:top]
        total_input = 0
        total_cache_read = 0
        has_cache_data = False
        model_name = ""
        provider = ""

        print(f"  Analyzing {func}/{variant} ({len(sample)} inferences)...", file=sys.stderr)

        for inf in sample:
            mi_data = api_get(f"{gateway}/internal/model_inferences/{inf['inference_id']}", headers)
            mis = mi_data.get("model_inferences", [])
            if not mis:
                continue
            mi = mis[0]
            model_name = mi.get("model_name", "")
            provider = mi.get("model_provider_name", "")
            input_tokens = mi.get("input_tokens") or 0
            cache_read = mi.get("provider_cache_read_input_tokens")

            total_input += input_tokens
            if cache_read is not None:
                has_cache_data = True
                total_cache_read += cache_read

        avg_input = total_input // len(sample) if sample else 0
        hit_rate = (total_cache_read / total_input * 100) if total_input > 0 and has_cache_data else None

        rows.append(
            {
                "function": func,
                "variant": variant,
                "model": model_name,
                "provider": provider,
                "count": len(sample),
                "avg_input": avg_input,
                "hit_rate": hit_rate,
                "total_input": total_input,
            }
        )

    # Sort by cache hit rate ascending (worst first), None values last
    rows.sort(key=lambda r: (r["hit_rate"] is not None, r["hit_rate"] or 0))

    # Print table
    print("\n## Prompt Cache Status\n")
    print("| Function | Variant | Model | Provider | Inferences | Avg Input Tokens | Cache Hit Rate |")
    print("|----------|---------|-------|----------|------------|-----------------|----------------|")

    for r in rows:
        rate_str = f"{r['hit_rate']:.0f}%" if r["hit_rate"] is not None else "No data"
        print(
            f"| {r['function']} | {r['variant']} | {r['model']} | {r['provider']} | {r['count']} | {r['avg_input']:,} | {rate_str} |"
        )

    # Summary
    total_pairs = len(rows)
    with_data = [r for r in rows if r["hit_rate"] is not None]
    good = [r for r in with_data if r["hit_rate"] >= 50]
    needs_attention = [r for r in with_data if r["hit_rate"] < 50]

    print("\n### Summary")
    print(f"- Total function+variant pairs: {total_pairs}")
    print(f"- Variants with cache hit rate >= 50%: {len(good)} / {len(with_data)}")
    print(f"- Variants needing attention (< 50%): {len(needs_attention)}")

    # Flag issues
    flagged = []
    for r in rows:
        if r["hit_rate"] is not None and r["hit_rate"] == 0 and r["avg_input"] > 1000:
            flagged.append(
                f"- **{r['function']}/{r['variant']}**: No caching detected ({r['avg_input']:,} avg input tokens). Run `/prompt-audit {r['function']} {r['variant']}` to investigate."
            )
        elif r["hit_rate"] is not None and r["hit_rate"] < 50 and "anthropic" in r["provider"]:
            flagged.append(
                f"- **{r['function']}/{r['variant']}**: Low cache hit rate ({r['hit_rate']:.0f}%). Consider adding cache breakpoints. Run `/prompt-audit {r['function']} {r['variant']}`."
            )
        elif r["hit_rate"] is None:
            flagged.append(f"- **{r['function']}/{r['variant']}**: No cache data available from provider.")

    if flagged:
        print("\n### Issues")
        for f in flagged:
            print(f)

    print()


if __name__ == "__main__":
    main()
