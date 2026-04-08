# CI Read-Only Provider Proxy Cache

## Problem

CI tests make live API calls to LLM providers (Anthropic, OpenAI, etc.) through a caching proxy.
This causes:

- **Flakiness**: Provider outages, rate limits, and non-deterministic responses break CI
- **Cost**: Every CI run burns API credits
- **Speed**: Network round-trips to providers add minutes to test suites
- **Security**: CI runners need provider API keys

The proxy already caches responses in Cloudflare R2, but today it falls through to live providers on cache miss.
We want CI to run **exclusively from cache** — no live API calls — while keeping a reliable way to populate and verify the cache.

## Background

### Provider Proxy

The provider-proxy is an HTTP/HTTPS MITM proxy (`crates/provider-proxy/`) that intercepts all provider API calls during e2e tests.
It caches request/response pairs as files on disk, keyed by a SHA256 hash of the sanitized request (headers, body).
The cache directory is synced to/from Cloudflare R2 between CI runs.

### Existing cache modes

| Mode                           | Behavior                                                            |
| ------------------------------ | ------------------------------------------------------------------- |
| `read-old-write-new` (default) | Serve cache hits; forward misses to provider and cache the response |
| `read-only`                    | Serve cache hits; forward misses to provider but don't cache        |
| `read-write`                   | Always read and write cache                                         |
| `read-only-require-hit`        | Serve cache hits; **fail with error** on miss (added in #7201)      |

### Current CI flow

1. Download cache from R2
2. Start provider-proxy in `read-old-write-new` mode
3. Run tests (cache hits served from disk, misses go to provider)
4. Upload cache back to R2 (merge queue and cron only)

### Aaron's test PR (#7205)

Switched all CI workflows to `read-only-require-hit` to measure cache coverage.
Results: most tests pass, but several categories fail due to non-deterministic cache keys.

## Failure Analysis

### Category 1: Non-deterministic request bodies

Tests embed random UUIDs, timestamps, or nonces in prompts.
Each run produces a different cache key.

**Examples:**

- `"If you see this, something has gone wrong: 019d641f-3802-75f3-..."` in prompt text
- Random episode IDs included in request context

**Affected tests:** `test_bad_auth_extra_headers`, `test_thinking_rejected_128k`, and others that include unique identifiers in LLM requests.

### Category 2: Random ports in URLs

Tests spin up local HTTP servers on random ports, then include image URLs like `https://127.0.0.1:35111/ferris.png` in requests.
The port changes every run, changing the cache key.

**Affected tests:** `test_image_url_with_fetch_false`, `test_forward_image_url`, `test_image_inference_store_filesystem`, all `test_image_inference_url*` client tests.

### Category 3: External resource fetches through the proxy

The gateway fetches images from `raw.githubusercontent.com` and these requests go through the proxy too.
The URL often includes a commit SHA that changes across branches.

**Affected tests:** `test_file_inference_url`, image tests referencing `ferris.png`.

### Category 4: Streaming responses

SSE streaming tests produce cache misses.
Root cause needs investigation — may be chunking differences or the proxy not caching streaming responses correctly.

**Affected tests:** `raw_response_streaming*`, `raw_usage_streaming*`, `openai_responses_tool_call_streaming`, `aggregated_response_chat_streaming`.

### Category 5: Extra headers and embeddings

Custom headers change the request signature. Some embedding provider responses aren't in the cache.

**Affected tests:** `test_extra_headers_raw`, `test_embeddings_cache_with_*_encoding`.

### Scale

| Test Suite   | Unique Failing Tests | Notes                                    |
| ------------ | -------------------- | ---------------------------------------- |
| live-tests   | ~7                   | 99.9% pass rate                          |
| client-tests | ~34                  | Mostly image/streaming/headers tests     |
| UI e2e       | ~1                   | Auth test with batch_writes config issue |

## Design

### Two-pass cache population with verification

The core idea: when the cache needs new entries, run the tests **twice** — once to write, once to verify.
This guarantees that what was written is deterministically reproducible.

#### Pass 1: Populate

- Run the target tests with `read-old-write-new` mode
- New cache entries are written to disk
- Record which tests ran (the test list)

#### Pass 2: Verify

- Run **exactly the same tests** from Pass 1
- Use `read-only-require-hit` mode
- Every request must hit cache — if any miss, **fail**
- A miss in Pass 2 means the request is non-deterministic (different cache key each run) and needs fixing

#### Why two passes?

A single write pass doesn't prove the cache works.
If a test includes a random UUID in the prompt, Pass 1 caches `hash(request-with-uuid-A)` but Pass 2 sends `hash(request-with-uuid-B)` and misses.
The verification pass catches non-determinism that a write-only pass would silently accept.

### Cache miss manifest

Add a new output to the provider-proxy: a `cache-misses.json` file that logs every cache miss during a run.

```json
[
  {
    "cache_key": "api.anthropic.com-733d4718b0c3...",
    "host": "api.anthropic.com",
    "method": "POST",
    "path": "/v1/messages",
    "timestamp": "2026-04-06T18:47:05Z"
  }
]
```

This file is useful for:

- Identifying which tests caused cache misses (correlate timestamps with test execution)
- Debugging non-deterministic cache keys (compare manifests between Pass 1 and Pass 2)
- Automated reporting in CI (post a summary comment on the PR)

### CI modes

| Trigger                      | Proxy Mode                   | Purpose                                    |
| ---------------------------- | ---------------------------- | ------------------------------------------ |
| PR / merge queue             | `read-only-require-hit`      | Strict: no live API calls                  |
| `/cache-populate` PR comment | Two-pass (write then verify) | Populate cache for new tests               |
| Cron / nightly               | `read-old-write-new`         | Refresh cache, detect provider API changes |

### `/cache-populate` workflow

```
Developer adds a new test that hits a provider
  → Pushes PR
  → CI fails: "Cache miss in ReadOnlyRequireHit mode"
  → Developer comments `/cache-populate` on PR

GitHub Actions workflow triggers:
  1. Download current cache from R2
  2. Pass 1: Run ALL tests in `read-old-write-new` mode
     - Existing cached tests pass as before
     - New tests hit the provider, responses are cached
  3. Upload cache to R2 (new entries added)
  4. Pass 2: Run ALL tests in `read-only-require-hit` mode
     - If green: cache is complete and deterministic
     - If red: post failure summary — non-deterministic requests detected
  5. If both passes green: mark the cache-populate check as passed
```

> **Open question:** Should Pass 1 and Pass 2 run ALL tests, or only the tests that failed in the initial CI run?
> Running all tests is simpler and more thorough (catches regressions).
> Running only failing tests is faster but requires parsing CI logs to extract the test list.
> **Recommendation:** Run all tests. The full suite takes ~30 minutes, and correctness matters more than speed for a manually-triggered workflow.

### Fixing non-deterministic tests

Before `/cache-populate` can work, we need to fix the sources of non-determinism.
These fixes should happen incrementally — each fix unblocks more tests for read-only mode.

#### Fix 1: Normalize random values in proxy cache key (proxy-side)

Extend the provider-proxy's request sanitization to strip or normalize:

- **UUIDv7 patterns** in JSON string values (`/[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}/`)
- **`127.0.0.1:<port>` and `localhost:<port>`** in URLs within request bodies → normalize to `127.0.0.1:0`

This is the highest-leverage fix: it makes the proxy tolerant of test non-determinism without changing test code.

#### Fix 2: Bypass proxy for non-provider hosts

Add `raw.githubusercontent.com` to the `no_proxy` list in the gateway's HTTP client setup (`crates/tensorzero-core/src/http.rs`).
These aren't provider API calls and don't need caching.

#### Fix 3: Investigate streaming cache misses

Determine why streaming tests miss cache. Possible causes:

- Streaming responses not being cached at all
- SSE chunk boundaries affecting the cached response format
- Request differences between streaming and non-streaming paths

#### Fix 4: Use deterministic values in tests (test-side, as needed)

For tests where proxy-side normalization isn't feasible, use fixed strings instead of random values:

- Use a fixed UUID constant instead of `Uuid::now_v7()` for prompt text that doesn't need uniqueness
- Use a fixed port for local test servers where possible

### Nightly cache refresh

The cron/nightly job continues running in `read-old-write-new` mode. This serves two purposes:

1. **Detect provider API changes**: If a provider changes their response format, the nightly run catches it
2. **Keep cache fresh**: Responses are updated to reflect current provider behavior

If a nightly run detects new failures, it should alert (e.g., Slack notification) so the team can investigate whether a provider changed their API.

### Cache invalidation

Cache entries become stale when:

- A provider changes their response format
- We update the model version in a test config
- We change the prompt/template used in a test

**Strategy:** The nightly `read-old-write-new` job naturally refreshes stale entries.
For intentional model/prompt changes, the developer uses `/cache-populate` to repopulate.

## Implementation Plan

### Phase 1: Proxy enhancements

1. Add cache miss manifest output (`cache-misses.json`) to provider-proxy
2. Add UUID normalization to proxy cache key sanitization
3. Add localhost port normalization to proxy cache key sanitization

### Phase 2: Fix test non-determinism

4. Add `raw.githubusercontent.com` to `no_proxy` list
5. Investigate and fix streaming cache misses
6. Fix remaining test-specific non-determinism as needed

### Phase 3: CI workflow changes

7. Add `/cache-populate` GitHub Actions workflow (two-pass)
8. Switch default CI mode to `read-only-require-hit`
9. Keep nightly cron in `read-old-write-new` mode

### Phase 4: Monitoring

10. Add Slack notification for nightly cache refresh failures
11. Add CI summary comment showing cache hit/miss statistics

## Related

- [#7201](https://github.com/tensorzero/tensorzero/pull/7201) — Added `read-only-require-hit` mode to provider-proxy (merged)
- [#7205](https://github.com/tensorzero/tensorzero/pull/7205) — [TEST] Check how many tests bypass cache (Aaron's test PR)
- `crates/provider-proxy/src/lib.rs` — Proxy implementation and cache modes
- `ci/download-provider-proxy-cache.sh` / `ci/upload-provider-proxy-cache.sh` — R2 sync scripts
- `crates/tensorzero-core/src/http.rs` — Gateway HTTP client proxy setup and `no_proxy` list
