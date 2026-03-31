#!/usr/bin/env python3
"""Prompt caching audit for TensorZero functions.

Queries the gateway API for recent inferences, checks tool ordering stability,
cache breakpoint configuration, and cache hit rates.
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
    limit = 100
    api_key = None
    function_name = None
    variant_name = None
    positional = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--gateway" and i + 1 < len(argv):
            gateway = argv[i + 1]
            i += 2
        elif arg == "--limit" and i + 1 < len(argv):
            limit = int(argv[i + 1])
            i += 2
        elif arg == "--api-key" and i + 1 < len(argv):
            api_key = argv[i + 1]
            i += 2
        else:
            positional.append(arg)
            i += 1

    if not positional:
        print(
            "Usage: prompt_audit.py <function_name> [variant_name] [--gateway URL] [--limit N] [--api-key KEY]",
            file=sys.stderr,
        )
        sys.exit(1)

    function_name = positional[0]
    if len(positional) > 1:
        variant_name = positional[1]

    # Resolve API key from env if not provided
    if not api_key:
        import os

        api_key = os.environ.get("TENSORZERO_API_KEY")

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return function_name, variant_name, gateway, limit, headers


def fetch_inferences(gateway, function_name, variant_name, limit, headers):
    body = {"function_name": function_name, "limit": limit}
    if variant_name:
        body["variant_name"] = variant_name
    data = api_post(f"{gateway}/v1/inferences/list_inferences", body, headers)
    return data.get("inferences", [])


def fetch_model_inference(gateway, inference_id, headers):
    data = api_get(f"{gateway}/internal/model_inferences/{inference_id}", headers)
    mis = data.get("model_inferences", [])
    return mis[0] if mis else None


def extract_tools_from_raw_request(raw_request_str):
    """Extract tool names from a raw_request JSON string."""
    try:
        raw = json.loads(raw_request_str)
    except (json.JSONDecodeError, TypeError):
        return []
    tools = raw.get("tools") or []
    names = []
    for t in tools:
        # Anthropic format: {"name": "...", ...}
        # OpenAI format: {"type": "function", "function": {"name": "...", ...}}
        if "function" in t and isinstance(t["function"], dict):
            names.append(t["function"].get("name", ""))
        else:
            names.append(t.get("name", ""))
    return names


def check_tool_ordering(model_inferences):
    """Check if tool definitions are stable across inferences using raw_request."""
    tool_signatures = set()
    has_tools = False
    for mi in model_inferences:
        names = extract_tools_from_raw_request(mi.get("raw_request", ""))
        sig = json.dumps(names)
        tool_signatures.add(sig)
        if names:
            has_tools = True

    if not has_tools:
        return "INFO", "No tools configured for this function."

    if len(tool_signatures) == 1:
        return "PASS", f"Tool definitions are stable across all {len(model_inferences)} inferences."
    else:
        return (
            "FAIL",
            f"Tool definitions appear in {len(tool_signatures)} different orders across {len(model_inferences)} inferences. Non-deterministic tool ordering breaks prefix caching.",
        )


def check_cache_breakpoints(model_inferences, provider):
    """Check if cache breakpoints are configured for Anthropic models."""
    is_anthropic = "anthropic" in provider.lower()

    if not is_anthropic:
        return "INFO", f"Provider `{provider}` uses automatic prefix caching — no explicit breakpoints needed."

    # Check raw_request for cache_control entries in system or tools
    has_breakpoints = False
    for mi in model_inferences:
        try:
            raw = json.loads(mi.get("raw_request", ""))
        except (json.JSONDecodeError, TypeError):
            continue
        # Check system blocks
        for block in raw.get("system", []):
            if isinstance(block, dict) and block.get("cache_control"):
                has_breakpoints = True
                break
        # Check tool definitions
        for tool in raw.get("tools", []):
            if isinstance(tool, dict) and tool.get("cache_control"):
                has_breakpoints = True
                break
        if has_breakpoints:
            break

    # Estimate system prompt length from model inference data
    sys_prompt_len = 0
    if model_inferences:
        system = model_inferences[0].get("system") or ""
        sys_prompt_len = len(system)

    estimated_tokens = sys_prompt_len / 4

    if has_breakpoints:
        return "PASS", "Cache breakpoints are configured via `extra_body`."

    if estimated_tokens > 1024:
        return (
            "FAIL",
            f"Anthropic model with long system prompt (~{int(estimated_tokens)} estimated tokens) but no `cache_control` breakpoints configured. Add breakpoints via `extra_body` to enable caching.",
        )
    else:
        return (
            "INFO",
            f"System prompt is short (~{int(estimated_tokens)} estimated tokens). Caching may not provide significant savings.",
        )


def check_cache_hit_rate(model_inferences):
    """Compute cache hit and write rates."""
    total_input = 0
    total_cache_read = 0
    total_cache_write = 0
    has_cache_data = False

    for mi in model_inferences:
        input_tokens = mi.get("input_tokens") or 0
        cache_read = mi.get("provider_cache_read_input_tokens")
        cache_write = mi.get("provider_cache_write_input_tokens")

        total_input += input_tokens
        if cache_read is not None:
            has_cache_data = True
            total_cache_read += cache_read
        if cache_write is not None:
            has_cache_data = True
            total_cache_write += cache_write

    if not has_cache_data:
        return None, None, total_input, total_cache_read, total_cache_write

    hit_rate = (total_cache_read / total_input * 100) if total_input > 0 else 0
    write_rate = (total_cache_write / total_input * 100) if total_input > 0 else 0
    return hit_rate, write_rate, total_input, total_cache_read, total_cache_write


def audit_variant(variant_name, inferences, gateway, headers):
    """Run all checks for a single variant."""
    # Sample up to 20 inferences for model-level data
    sample = inferences[:20]
    model_inferences = []
    for inf in sample:
        mi = fetch_model_inference(gateway, inf["inference_id"], headers)
        if mi:
            model_inferences.append(mi)

    if not model_inferences:
        return None

    model_name = model_inferences[0].get("model_name", "unknown")
    provider = model_inferences[0].get("model_provider_name", "unknown")

    # Run checks
    tool_status, tool_detail = check_tool_ordering(model_inferences)
    bp_status, bp_detail = check_cache_breakpoints(model_inferences, provider)
    hit_rate, write_rate, total_input, total_read, total_write = check_cache_hit_rate(model_inferences)

    return {
        "variant": variant_name,
        "model": model_name,
        "provider": provider,
        "sample_size": len(model_inferences),
        "tool_ordering": (tool_status, tool_detail),
        "cache_breakpoints": (bp_status, bp_detail),
        "hit_rate": hit_rate,
        "write_rate": write_rate,
        "total_input": total_input,
        "total_cache_read": total_read,
        "total_cache_write": total_write,
    }


def print_report(function_name, results):
    """Print the final audit report."""
    for r in results:
        print(f"\n## Cache Audit: {function_name} / {r['variant']}")
        print(f"**Model:** {r['model']} (provider: {r['provider']})")
        print(f"**Sample size:** {r['sample_size']} inferences")

        # Tool ordering
        ts, td = r["tool_ordering"]
        print(f"\n### Tool Ordering: {ts}")
        print(td)

        # Cache breakpoints
        bs, bd = r["cache_breakpoints"]
        print(f"\n### Cache Breakpoints: {bs}")
        print(bd)

        # Cache hit rate
        if r["hit_rate"] is None:
            print("\n### Cache Hit Rate: No data")
            print("Provider does not report cache statistics.")
        else:
            print(f"\n### Cache Hit Rate: {r['hit_rate']:.1f}%")
            print(f"- Total input tokens: {r['total_input']:,}")
            print(f"- Cache read tokens: {r['total_cache_read']:,}")
            print(f"- Cache write tokens: {r['total_cache_write']:,}")
            if r["hit_rate"] < 50 and r["total_input"] > 5000:
                print("- **Cache hit rate is low.** Check breakpoint configuration and system prompt stability.")

        # Recommendations
        recommendations = []
        if r["tool_ordering"][0] == "FAIL":
            recommendations.append("Fix non-deterministic tool ordering — ensure tools are defined in a stable order.")
        if r["cache_breakpoints"][0] == "FAIL":
            recommendations.append(
                "Add `cache_control` breakpoints via `extra_body` in the variant config, e.g.:\n"
                '  `{ pointer = "/system/0/cache_control", value = { type = "ephemeral" } }`'
            )
        if r["hit_rate"] is not None and r["hit_rate"] == 0 and r["total_input"] > 5000:
            if r["cache_breakpoints"][0] != "FAIL":
                recommendations.append(
                    "Cache hit rate is 0% despite significant input tokens. Verify cache breakpoints are placed correctly."
                )

        if recommendations:
            print("\n### Recommendations")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("\n### Recommendations")
            print("None — this variant is well-configured for prompt caching.")

    print()


def main():
    function_name, variant_name, gateway, limit, headers = parse_args(sys.argv[1:])

    print(f"Fetching inferences for `{function_name}`...", file=sys.stderr)
    inferences = fetch_inferences(gateway, function_name, variant_name, limit, headers)

    if not inferences:
        print(
            f"\nNo inferences found for function `{function_name}`"
            + (f" variant `{variant_name}`" if variant_name else "")
            + ". Run some inferences first."
        )
        sys.exit(0)

    # Group by variant
    by_variant = defaultdict(list)
    for inf in inferences:
        by_variant[inf["variant_name"]].append(inf)

    print(f"Found {len(inferences)} inferences across {len(by_variant)} variant(s).", file=sys.stderr)

    results = []
    for vname, vinfs in sorted(by_variant.items()):
        print(f"Auditing `{vname}` ({len(vinfs)} inferences)...", file=sys.stderr)
        r = audit_variant(vname, vinfs, gateway, headers)
        if r:
            results.append(r)

    print_report(function_name, results)


if __name__ == "__main__":
    main()
