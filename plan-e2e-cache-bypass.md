# Plan: E2E Tests Cache Bypass Analysis and Remediation

## Issue Summary

GitHub Issue #5380 addresses provider-side changes blocking merge queues. The team has agreed to:
1. Use only cached responses in merge queue tests for deterministic CI
2. Run live tests conditionally/periodically, not blocking merges
3. Discontinue cache-bypassing for most tests

## Current Caching Architecture

### Two-Layer Cache System

1. **Gateway Cache (ClickHouse-backed)**:
   - Location: `tensorzero-core/src/cache.rs`
   - Stores inference responses in ClickHouse
   - Controlled by `CacheEnabledMode`: `On`, `Off`, `ReadOnly`, `WriteOnly`
   - Uses BLAKE3 hash of (model_name + provider_name + request_json) as cache key

2. **Provider-Proxy Cache (Disk-based)**:
   - Location: `provider-proxy/src/lib.rs`
   - HTTP response cache for provider requests
   - Uses SHA256 hash of serialized HTTP request as cache key
   - Modes: `ReadOnly`, `ReadWrite`, `ReadOldWriteNew`
   - Default mode: `ReadOldWriteNew` (reads old entries, rewrites on miss)
   - Cache stored in R2 and synced during CI
   - **Important**: Error responses are NOT cached by provider-proxy

### CI Pipeline

- **Merge Queue**: Downloads provider-proxy cache from R2, runs e2e tests, uploads cache on success
- **Scheduled Cron**: Runs without cache to detect provider issues
- Cache header: `x-tensorzero-provider-proxy-cache: true/false`

---

## Implementation Summary

### Changes Made

#### 1. Raw Usage Tests - Fixed (6 tests)

**File: `tensorzero-core/tests/e2e/raw_usage/cache.rs`**

Changed from dynamic UUIDs to fixed test-specific strings for deterministic provider-proxy caching:

| Test | Before | After |
|------|--------|-------|
| `test_raw_usage_cache_behavior_non_streaming` | `Uuid::now_v7()` | Fixed string `..._v1` |
| `test_raw_usage_cache_behavior_streaming` | `Uuid::now_v7()` | Fixed string `..._v1` |
| `test_raw_usage_cache_disabled` | `Uuid::now_v7()` | Fixed string `..._v1` |
| `test_raw_usage_cache_disabled_streaming` | `Uuid::now_v7()` | Fixed string `..._v1` |
| `test_raw_usage_cache_openai_compatible_non_streaming` | `Uuid::now_v7()` | Fixed string `..._v1` |
| `test_raw_usage_cache_openai_compatible_streaming` | `Uuid::now_v7()` | Fixed string `..._v1` |

**Rationale**: These tests expect SUCCESS responses which ARE cached by provider-proxy. Using fixed strings allows cache hits across CI runs. Gateway cache (ClickHouse) is fresh each CI run, so first request within a test will still be a cache miss.

#### 2. Provider Error Tests - Documented

**File: `tensorzero-core/tests/e2e/providers/anthropic.rs`**

- `test_thinking_rejected_128k`: Added documentation explaining this test expects an error response (BAD_GATEWAY), which provider-proxy does NOT cache. The test will always hit live providers.

**File: `tensorzero-core/tests/e2e/providers/common.rs`**

- `test_bad_auth_extra_headers_with_provider_and_stream`: Added documentation explaining this test expects auth error responses, which provider-proxy does NOT cache.

**Rationale**: Error responses are not cached by provider-proxy, so these tests cannot benefit from caching. They should be moved to periodic live tests if they cause merge queue instability.

#### 3. Cache Tests - No Changes Needed

**File: `tensorzero-core/tests/e2e/cache.rs`**

These tests use dummy providers (`model = "test"`, `model = "dummy::*"`), so they don't hit live providers. The random seeds are for gateway cache uniqueness, not provider-proxy cache bypass.

#### 4. Evaluation Tests - No Changes Needed

**File: `tensorzero-core/tests/e2e/endpoints/internal/evaluations.rs`**

These tests use dummy providers (`variant_name: "test"` which uses `model = "test"`). The `inference_cache: "off"` only disables gateway cache, not provider-proxy cache.

#### 5. CI Configuration - Updated

**File: `ci/run-provider-proxy.sh`**

Added support for `PROVIDER_PROXY_CACHE_MODE` environment variable:
- Default: `read-old-write-new` (current behavior)
- Options: `read-old-write-new`, `read-only`, `read-write`

**File: `tensorzero-core/tests/e2e/docker-compose.live.yml`**

Added support for `PROVIDER_PROXY_CACHE_MODE` environment variable in docker-compose command.

---

## Tests That Still Hit Live Providers

These tests expect error responses which are NOT cached by provider-proxy:

| File | Test | Reason |
|------|------|--------|
| `providers/anthropic.rs` | `test_thinking_rejected_128k` | Expects BAD_GATEWAY error |
| `providers/common.rs` | `test_bad_auth_extra_headers_with_provider_and_stream` | Expects auth errors |

**Recommendation**: Move these to periodic live tests or skip in merge queue to avoid blocking on provider issues.

---

## Future Work

1. **Enable ReadOnly Mode for Merge Queue**: Set `PROVIDER_PROXY_CACHE_MODE=read-only` in merge queue CI to prevent cache misses from hitting live providers. Tests will fail if cache is missing (good - forces explicit cache regeneration).

2. **Create Cache Regeneration Workflow**: Manual trigger to refresh provider-proxy cache by running all tests with live providers and uploading to R2.

3. **Move Error Tests to Periodic**: Create a separate workflow for tests that expect error responses, running periodically with Slack alerts instead of blocking merge queue.

---

## Summary

| Category | Test Count | Action Taken |
|----------|------------|--------------|
| Raw usage tests with UUIDs | 6 | ✅ Changed to fixed strings |
| Provider error tests | 2 | ✅ Documented (can't cache errors) |
| Cache tests with random seeds | 6 | ✅ No changes (use dummy providers) |
| Evaluation tests | 5 | ✅ No changes (use dummy providers) |
| CI configuration | - | ✅ Added configurable cache mode |
