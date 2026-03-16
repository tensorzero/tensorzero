# RFC: Prompt Caching Optimization Detector

## Problem

When users run many inferences through the same TensorZero variant, there is often significant overlap in the content sent to the model provider (system prompts, tool definitions, few-shot examples, shared context). Anthropic's prompt caching can reduce input token costs by up to **90%** and latency by up to **85%** for cached prefixes — but users currently have no way to know if they're leaving money on the table.

We want to build:

1. **A diagnostic algorithm** that analyzes ~100 recent inferences for a variant and reports caching opportunities.
2. **An Autopilot Cookbook** that wraps this into a one-click audit.

## Background: How Prompt Caching Works

Prompt caching is **prefix-based**. The cache hierarchy is:

```
tools → system → messages
```

The API hashes content sequentially. A cache hit requires that all content from the start up to a breakpoint is **byte-identical** to a previous request. Any change at level N invalidates levels N and beyond.

Key constraints:
- **Min token thresholds**: 1,024–4,096 tokens depending on model
- **TTL**: 5 minutes (default), 1 hour (2x base cost)
- **Max 4 breakpoints** per request
- **Cache reads cost 0.1x** base input price
- **Cache writes cost 1.25x** base input price

## Algorithm

### Input

Given **N** inferences (e.g., 100) for a specific `(function_name, variant_name)` pair:

```
For each inference i ∈ [1, N]:
  - raw_request_i:  JSON string sent to the provider API
  - input_tokens_i: total input tokens (from DB)
  - created_at_i:   timestamp
  - model_name_i:   model used
```

### Phase 1: Structural Decomposition

Parse each `raw_request` into **ordered segments** following the cache hierarchy:

```
S_tools(i)   = canonical JSON string of raw_request_i["tools"]     (or "" if absent)
S_system(i)  = canonical JSON string of raw_request_i["system"]    (or "" if absent)
S_msgs(i)    = canonical JSON string of raw_request_i["messages"]
```

Concatenate into a single cache-ordered string:

```
S(i) = S_tools(i) + "||" + S_system(i) + "||" + S_msgs(i)
```

The `||` delimiters ensure we don't accidentally create false prefix matches across segment boundaries.

### Phase 2: Segment Stability Analysis

For each segment type, compute **stability** — the fraction of request pairs where the segment is identical:

```
stability(segment) = count(S_segment(i) == S_segment(i-1) for i in [2,N]) / (N - 1)
```

Also compute per-segment **unique values**:

```
uniqueness(segment) = |{S_segment(i) : i ∈ [1,N]}| / N
```

| Metric | Interpretation |
|--------|---------------|
| `stability(tools) = 1.0` | Tools never change (expected — they come from config) |
| `stability(system) = 0.95` | System prompt is mostly static, minor template variation |
| `stability(system) = 0.3` | System prompt varies heavily (e.g., date injection, user-specific context) |
| `uniqueness(messages) = 1.0` | Every request has unique messages (expected for single-turn) |

### Phase 3: Longest Common Prefix (LCP) Analysis

For each consecutive pair `(i, i+1)`, compute the **longest common prefix** of `S(i)` and `S(i+1)`:

```
lcp_chars(i) = length of longest common prefix of S(i) and S(i+1)
total_chars(i) = length of S(i)
prefix_ratio(i) = lcp_chars(i) / total_chars(i)
```

Aggregate metrics:

```
mean_prefix_ratio  = mean(prefix_ratio(i) for i in [1, N-1])
p25_prefix_ratio   = 25th percentile
p75_prefix_ratio   = 75th percentile
min_prefix_ratio    = min
```

> **Why character-level, not token-level?**
> An identical character prefix guarantees an identical token prefix (tokenization is deterministic and prefix-preserving). Character comparison is O(n) and needs no tokenizer. We use the known `input_tokens` to convert character ratios to token estimates.

### Phase 4: Token Estimation

Estimate cacheable tokens per request:

```
chars_per_token(i) = total_chars(i) / input_tokens_i
cacheable_tokens(i) = lcp_chars(i) / chars_per_token(i)
uncached_tokens(i) = input_tokens_i - cacheable_tokens(i)
```

Cross-validate: if `chars_per_token` varies wildly across requests (e.g., due to images/documents), fall back to a fixed estimate of **4 chars/token** for text and flag that binary content is present.

Aggregate:

```
mean_cacheable_tokens = mean(cacheable_tokens(i))
mean_cacheable_pct    = mean(cacheable_tokens(i) / input_tokens_i)
```

### Phase 5: Temporal Analysis (Cache Hit Probability)

Cache has a TTL. Even if prefixes overlap perfectly, requests must arrive within the TTL window.

```
intervals(i) = created_at(i+1) - created_at(i)    for consecutive requests
```

Compute:

```
p50_interval = median(intervals)
p90_interval = 90th percentile
pct_within_5min = count(intervals < 5 min) / len(intervals)
pct_within_1hr  = count(intervals < 1 hr) / len(intervals)
```

**Estimated cache hit rate** under each TTL:

```
hit_rate_5min = pct_within_5min * mean_prefix_ratio
hit_rate_1hr  = pct_within_1hr * mean_prefix_ratio
```

This is a lower bound — in practice, non-consecutive requests can also hit the cache if they share the same prefix and fall within the TTL.

### Phase 6: Cost Savings Estimation

Pricing model (per token):

| Scenario | Cached tokens cost | Uncached tokens cost |
|----------|-------------------|---------------------|
| No caching | `base_price` | `base_price` |
| Cache write (miss) | `1.25 × base_price` | `base_price` |
| Cache read (hit) | `0.1 × base_price` | `base_price` |

Base prices per million input tokens:

```typescript
const BASE_INPUT_PRICES: Record<string, number> = {
  "claude-opus-4-6":   15.00,
  "claude-opus-4-5":   15.00,
  "claude-sonnet-4-6":  3.00,
  "claude-sonnet-4-5":  3.00,
  "claude-haiku-4-5":   0.80,
  // ...
};
```

For each request, estimate cost under three scenarios:

```
// Current cost (no caching)
cost_nocache(i) = input_tokens(i) * base_price

// With caching — cache miss (write)
cost_miss(i) = cacheable_tokens(i) * 1.25 * base_price
             + uncached_tokens(i) * base_price

// With caching — cache hit (read)
cost_hit(i) = cacheable_tokens(i) * 0.1 * base_price
            + uncached_tokens(i) * base_price
```

Expected cost with 5-minute TTL:

```
expected_cost_5min(i) = hit_rate_5min * cost_hit(i) + (1 - hit_rate_5min) * cost_miss(i)
```

Aggregate savings:

```
total_current_cost      = sum(cost_nocache(i))
total_expected_cost_5min = sum(expected_cost_5min(i))
savings_pct_5min        = 1 - total_expected_cost_5min / total_current_cost

// Same for 1-hour TTL (using 2x write cost instead of 1.25x)
```

### Phase 7: Shuffle Detection

Prompt caching is prefix-based, so **reordering content kills cache hits** even if the content is identical. We detect this:

For the **messages** segment, extract individual message content blocks and check if the same blocks appear across requests but in different orders:

```
blocks(i) = set of (role, content_hash) tuples from messages in request i
```

For each pair `(i, i+1)`:

```
jaccard(i) = |blocks(i) ∩ blocks(i+1)| / |blocks(i) ∪ blocks(i+1)|
sequence_match(i) = prefix_ratio for the messages segment only
```

**Shuffle signal** = high Jaccard similarity (content overlap) but low sequence match (different ordering):

```
shuffle_score(i) = jaccard(i) - sequence_match(i)
mean_shuffle_score = mean(shuffle_score(i))
```

| `mean_shuffle_score` | Interpretation |
|---------------------|---------------|
| < 0.05 | No shuffling detected |
| 0.05 – 0.3 | Mild shuffling — some content reordered |
| > 0.3 | Significant shuffling — reordering would unlock caching |

### Phase 8: Where to Place Breakpoints

Based on the analysis, recommend breakpoint placement:

```
Candidates (in cache hierarchy order):
1. After tools     — if stability(tools) > 0.9 and token_count(tools) > min_threshold
2. After system    — if stability(system) > 0.9 and cumulative tokens > min_threshold
3. After message N — if the first N messages are stable across requests
```

For each candidate, estimate the **incremental benefit**: how many additional tokens become cacheable.

Emit at most **4 recommendations** (API limit), prioritizing by incremental benefit.

### Phase 9: Minimum Token Threshold Check

Verify the cacheable prefix meets the model's minimum:

```typescript
const MIN_CACHE_TOKENS: Record<string, number> = {
  "claude-opus-4-6": 4096,
  "claude-opus-4-5": 4096,
  "claude-sonnet-4-6": 2048,
  "claude-sonnet-4-5": 1024,
  "claude-sonnet-4": 1024,
  "claude-haiku-4-5": 4096,
  "claude-haiku-3-5": 2048,
};
```

If `mean_cacheable_tokens < min_threshold`, flag that caching is not applicable even though overlap exists.

## Output: Diagnostic Report

```typescript
interface PromptCachingReport {
  function_name: string;
  variant_name: string;
  model_name: string;
  sample_size: number;
  time_range: { from: string; to: string };

  // Stability
  segment_stability: {
    tools: number;      // 0-1
    system: number;     // 0-1
    messages: number;   // 0-1 (first message position only)
  };

  // Prefix analysis
  prefix_analysis: {
    mean_prefix_ratio: number;       // 0-1, fraction of request that's a common prefix
    p25_prefix_ratio: number;
    p75_prefix_ratio: number;
    mean_cacheable_tokens: number;
    mean_uncached_tokens: number;
    meets_min_threshold: boolean;
  };

  // Temporal
  temporal: {
    p50_interval_seconds: number;
    pct_within_5min: number;
    pct_within_1hr: number;
    estimated_hit_rate_5min: number;
    estimated_hit_rate_1hr: number;
  };

  // Cost
  cost_estimate: {
    current_cost_per_1k_requests: number;
    estimated_cost_5min_ttl: number;
    estimated_cost_1hr_ttl: number;
    savings_pct_5min: number;
    savings_pct_1hr: number;
    recommended_ttl: "5min" | "1hr" | "none";
  };

  // Shuffle
  shuffle_detection: {
    mean_shuffle_score: number;
    shuffle_detected: boolean;
    recommendation: string | null;  // e.g. "Reorder messages to place static content first"
  };

  // Breakpoint recommendations
  breakpoint_recommendations: Array<{
    position: string;             // e.g. "after system[0]", "after messages[2]"
    extra_body_pointer: string;   // e.g. "/system/0/cache_control"
    extra_body_value: object;     // e.g. {"type": "ephemeral"}
    estimated_cacheable_tokens: number;
    incremental_savings_pct: number;
  }>;

  // Overall
  opportunity_score: number;  // 0-100, composite score
  summary: string;            // Human-readable one-liner
}
```

### Opportunity Score

Composite score (0–100) combining the key factors:

```
opportunity_score =
    w1 * mean_cacheable_pct              // How much content is cacheable (0-1)
  * w2 * estimated_hit_rate              // How often cache would be hit (0-1)
  * w3 * normalize(savings_pct)          // How much money is saved (0-1)
  * w4 * (1 if meets_min_threshold else 0)  // Binary gate

where w1=0.3, w2=0.3, w3=0.3, w4=0.1 (tunable)
```

Scaled to 0–100 and thresholded:

| Score | Label |
|-------|-------|
| 80–100 | **High opportunity** — enable caching immediately |
| 50–79 | **Moderate opportunity** — consider caching, review recommendations |
| 20–49 | **Low opportunity** — caching possible but savings are small |
| 0–19 | **No opportunity** — prefix overlap too small or requests too infrequent |

## Autopilot Cookbook: "Audit Prompt Caching Opportunities"

### Cookbook Concept

A Cookbook is a **predefined Autopilot script** — a sequence of tool calls with a fixed system prompt that performs a specific audit or optimization task. The user launches it from the UI, it runs autonomously, and produces a report.

### Flow

```
1. User selects "Audit Prompt Caching" cookbook in Autopilot UI
2. User picks target function (or "all functions")
3. Cookbook session starts with system prompt:

   "You are an optimization auditor. For each function/variant pair,
    retrieve the last 100 inferences and analyze prompt caching opportunities.
    Use the prompt_caching_analyze tool to run the analysis.
    Present findings sorted by opportunity_score descending.
    For high-opportunity variants, propose concrete extra_body config changes."

4. Autopilot calls:
   - list_inferences(function_name=X, variant_name=Y, limit=100)
   - prompt_caching_analyze(inferences=<result>)    ← NEW TOOL
   - Repeats for each variant

5. Autopilot produces:
   - Summary table of all variants with opportunity scores
   - Detailed report for top opportunities
   - Ready-to-apply config changes (extra_body with cache_control pointers)
```

### New Autopilot Tool: `prompt_caching_analyze`

```rust
/// Analyzes a batch of inferences for prompt caching optimization opportunities.
///
/// Input: list of (raw_request JSON, input_tokens, created_at, model_name)
/// Output: PromptCachingReport
struct PromptCachingAnalyzeTool;
```

This tool implements the algorithm described above. It runs entirely in-memory (no DB access needed — the data is passed in from `list_inferences`).

### Alternative: Standalone Script

For users who want to run this outside Autopilot, provide a standalone TypeScript script:

```bash
npx tensorzero-prompt-cache-audit \
  --gateway-url http://localhost:3000 \
  --function my_function \
  --variant my_variant \
  --limit 100
```

## Edge Cases & Limitations

1. **Binary content (images, documents)**: Character-level prefix comparison works but `chars_per_token` estimation breaks down. Fall back to treating binary content as opaque blocks — compare by hash, not by character prefix.

2. **Multi-provider variants**: If a variant routes to different providers, each provider has different caching semantics. Group analysis by `model_provider_name`.

3. **Dynamic tools**: If tool definitions change between requests (e.g., dynamic tools injected per-request), `stability(tools)` will be low and the entire cache hierarchy is invalidated. This is an important finding to surface.

4. **Extra body already has cache_control**: Check if `extra_body` in the inference already contains `cache_control` entries. If so, report current caching effectiveness rather than "enable caching" recommendations.

5. **Non-Anthropic providers**: OpenAI and other providers have different (or no) caching APIs. This analysis is initially **Anthropic-specific**. Flag non-Anthropic variants as "not applicable".

6. **Streaming vs non-streaming**: Caching works the same for both. No special handling needed.

## Cross-Provider Considerations

### OpenAI

OpenAI has **automatic** prefix caching (no opt-in required):
- Activates for prompts >1,024 tokens
- Cached tokens returned in `usage.prompt_tokens_details.cached_tokens`
- Cached tokens are **50% cheaper** (not 90% like Anthropic)
- Cache is at the **organization level** — requests from different users in the same org share cache
- No explicit breakpoints — the system automatically caches the longest matching prefix
- Tool definitions and their **ordering** must remain identical to be included in the cached prefix

**Implications for our algorithm**: OpenAI users benefit from the same prefix stability analysis, but the cost model differs (50% vs 90% savings) and there are no breakpoint recommendations to make — only content ordering recommendations.

### Google Gemini

Gemini has two mechanisms:
- **Implicit caching** (automatic, like OpenAI) — enabled by default
- **Explicit caching** — user creates a named cache object with a configurable TTL (default 1 hour, no min/max bounds)
- Min thresholds: 1,024–4,096 tokens depending on model
- Response includes `usage_metadata` with cache hit token counts

**Implications**: Gemini's explicit caching with configurable TTL means we could recommend creating named caches for highly stable prefixes, not just ephemeral breakpoints.

### Provider-Agnostic Recommendations

The core algorithm (prefix stability, temporal analysis, shuffle detection) applies across all providers. The output layer should be provider-specific:

| Provider | Breakpoint control | Savings multiplier | TTL options |
|----------|-------------------|-------------------|-------------|
| Anthropic | Explicit (max 4) + automatic | 0.1x read, 1.25x write | 5min, 1hr |
| OpenAI | Automatic only | 0.5x read | Auto-eviction |
| Gemini | Explicit named caches | Varies | Configurable |

## Insights from Inference Engine Research

### SGLang RadixAttention

SGLang (LMSYS) uses a **radix tree** to manage KV cache at the inference engine level. Key ideas relevant to our analysis:

1. **Radix tree for prefix matching**: Edges are labeled with token sequences of varying length. This allows O(prefix_length) lookup for the longest matching cached prefix. We don't need a radix tree for our offline analysis (we have ~100 requests, not millions), but the concept validates that **prefix-based matching is the correct abstraction**.

2. **LRU eviction with cache-aware scheduling**: SGLang orders requests to maximize cache reuse — requests with similar prefixes are batched together. This is analogous to our temporal analysis: if a user's requests arrive in clusters with shared prefixes, cache hit rates are higher.

3. **Automatic prefix detection**: SGLang's runtime does prefix matching transparently — "the front end always sends full prompts to the runtime and the runtime will automatically do prefix matching." This mirrors OpenAI's approach. Our tool adds value by making this invisible optimization **visible** so users can restructure prompts to maximize it.

### Prompt Modules (Gim et al., 2023)

The "Prompt Cache" paper proposes **prompt modules** — explicitly defined reusable text segments with positional encoding preservation. Key insight: they achieve 8x speedup on GPU by precomputing attention states for common prompt components.

**Relevance to our work**: This validates decomposing prompts into segments (tools, system, message blocks) and measuring segment stability independently. The segments that are most stable are the best candidates for caching.

## Refined Algorithm: Key Improvements from Research

Based on the research, several refinements to the original algorithm:

### Improvement 1: Block-Level Granularity (not just segment-level)

Instead of comparing entire segments (`system`, `messages`), decompose into **individual content blocks** and track stability per block position. This catches cases like:

```
Request 1: system = [block_A, block_B, block_C]
Request 2: system = [block_A, block_B, block_D]  ← block_C changed, but A+B are cacheable
```

The breakpoint recommendation should be "place `cache_control` on `system[1]`" (after block_B).

### Improvement 2: Invalidation Cascade Awareness

From the Anthropic docs, changing certain parameters invalidates the cache at different levels:

| Change | Invalidates from |
|--------|-----------------|
| Tool definitions changed | tools level (everything) |
| `tool_choice` changed | system level onward |
| Images added/removed | system level onward |
| Thinking parameters changed | system level onward |
| `speed` setting changed | system level onward |

Our analysis should check for these **hidden invalidators** across requests, not just content changes. For example, if `tool_choice` flips between `auto` and `required` across requests, that kills the system+message cache even if system content is identical.

### Improvement 3: Concurrent Request Handling

From the Anthropic docs: cache is only available **after the first response begins streaming**. If two requests are sent in parallel, the second may not hit the cache. Our temporal analysis should detect **burst patterns** (multiple requests within <1 second) and flag that concurrent requests may not benefit from caching even with perfect prefix overlap.

```
burst_count = count of intervals < 1 second
burst_ratio = burst_count / (N - 1)
```

If `burst_ratio > 0.1`, add a warning: "Concurrent requests detected — consider sequencing requests to allow cache population."

### Improvement 4: Mixed TTL Recommendation

When different parts of the prompt change at different frequencies, recommend **mixed TTLs**:

```
If stability(system) > 0.95 AND p50_system_change_interval > 1 hour:
  → Recommend 1-hour TTL on system breakpoint
If stability(first_N_messages) > 0.8 AND p50_message_change_interval < 5 min:
  → Recommend 5-min TTL on message breakpoint
```

This is supported by Anthropic's API (different `cache_control.ttl` per breakpoint) and minimizes write costs for rarely-changing content.

### Improvement 5: 20-Block Lookback Window Awareness

From the Anthropic docs: the cache checker only looks back **20 blocks** from each breakpoint. If a prompt has >20 content blocks and only one breakpoint at the end, changes to early blocks won't be detected as cache misses (the check stops after 20 blocks).

Our breakpoint recommendation should:
1. Count total content blocks across all segments
2. If total > 20, **require** intermediate breakpoints every ~15-20 blocks
3. Place breakpoints strategically before content that changes frequently

### Improvement 6: JSON Key Ordering Stability

From the Anthropic and OpenAI docs: cache matching is byte-level. If JSON serialization produces different key orderings across requests (common in Go, Swift, and some Python configurations), cache hits fail even though the semantic content is identical.

Our analysis should detect this:
```
For each pair (i, i+1) where prefix_ratio is low but Jaccard similarity is high:
  Parse both as JSON and compare with sorted keys
  If sorted comparison matches but raw doesn't → flag JSON key ordering issue
```

## Open Questions

1. **Should we compare ALL pairs or just consecutive pairs?** Consecutive pairs are O(N) and represent the most likely cache reuse pattern. All-pairs is O(N²) but catches patterns where requests from different users share prefixes. **Recommendation: consecutive pairs by default, with an option for all-pairs.**

2. **Should we use actual tokenization?** Using `tiktoken` or similar would give exact token counts but adds a dependency and is slower. The character-ratio approach is fast and accurate enough for an estimator. **Recommendation: character-level for v1, optional tokenizer for v2.**

3. **How to handle the 20-block lookback limit?** The API only checks up to 20 content blocks backwards from the breakpoint. For long conversations (>20 messages), we should recommend placing explicit breakpoints before block 20. **Recommendation: include this in breakpoint recommendations.**

4. **Per-user vs aggregate analysis?** If different users get different system prompts (via template variables), aggregate analysis might undercount opportunities. **Recommendation: v1 does aggregate analysis, v2 could group by template variable hash.**

## References

- [Anthropic Prompt Caching Docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [OpenAI Prompt Caching Guide](https://developers.openai.com/cookbook/examples/prompt_caching101)
- [Google Gemini Context Caching](https://ai.google.dev/gemini-api/docs/caching)
- [SGLang RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/) — Radix tree for KV cache management with LRU eviction
- [Prompt Cache: Modular Attention Reuse (Gim et al., 2023)](https://arxiv.org/abs/2311.04934) — Prompt modules with precomputed attention states
- [Anthropic Blog: Prompt Caching Announcement](https://claude.com/blog/prompt-caching) — Real-world savings: 79% latency reduction, 90% cost savings for 100K-token book chats
