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

### CI Pipeline

- **Merge Queue**: Downloads provider-proxy cache from R2, runs e2e tests, uploads cache on success
- **Scheduled Cron**: Runs without cache to detect provider issues
- Cache header: `x-tensorzero-provider-proxy-cache: true/false`

---

## Tests That Bypass Cache

### Category 1: Randomized Seeds/Content (Provider-Proxy Cache Bypass)

These tests inject randomness to ensure they hit live providers, not cached responses.

| File | Test | Line | Bypass Method | Purpose |
|------|------|------|---------------|---------|
| `e2e/cache.rs` | `test_cache_write_and_read` | 63 | `rand::random::<u32>()` in seed | Test cache write/read cycle |
| `e2e/cache.rs` | `test_cache_stream_write_and_read` | 194 | `rand::random::<u32>()` in seed | Test streaming cache write/read |
| `e2e/cache.rs` | `test_streaming_cache_with_err` | 478 | `rand::rng().random_range()` | Test error handling prevents cache write |
| `e2e/cache.rs` | `test_streaming_cache_without_err` | 492 | `rand::rng().random_range()` | Test successful streaming cache |
| `e2e/cache.rs` | `test_dont_cache_invalid_tool_call` | 368 | `Uuid::now_v7()` in message | Test cache skip on invalid tools |
| `e2e/cache.rs` | `test_dont_cache_tool_call_schema_error` | 418 | `Uuid::now_v7()` in message | Test cache skip on schema error |
| `e2e/providers/anthropic.rs` | `test_thinking_rejected_128k` | 186 | `Uuid::now_v7()` in message | Test current Anthropic behavior |
| `e2e/providers/common.rs` | `test_bad_auth_extra_headers_with_provider_and_stream` | 2333 | `Uuid::now_v7()` in message | Test auth error handling |

### Category 2: Explicit Gateway Cache Disable

| File | Test | Line | Setting | Purpose |
|------|------|------|---------|---------|
| `e2e/raw_usage/cache.rs` | `test_raw_usage_cache_disabled` | 207 | `"enabled": "off"` | Test raw_usage without cache |
| `e2e/raw_usage/cache.rs` | `test_raw_usage_cache_disabled_streaming` | 251 | `"enabled": "off"` | Streaming variant |
| `e2e/raw_usage/cache.rs` | `test_raw_usage_cache_openai_compatible_non_streaming` | 396 | `"enabled": "off"` | OpenAI-compatible variant |
| `e2e/raw_usage/cache.rs` | `test_raw_usage_cache_openai_compatible_streaming` | 426 | `"enabled": "off"` | OpenAI-compatible streaming |

### Category 3: Evaluation Tests (Explicit Cache Disable)

| File | Test | Line | Setting | Purpose |
|------|------|------|---------|---------|
| `e2e/endpoints/internal/evaluations.rs` | `test_run_evaluation_streaming_success` | 713 | `"inference_cache": "off"` | Evaluations need fresh inferences |
| `e2e/endpoints/internal/evaluations.rs` | `test_run_evaluation_streaming_missing_variant` | 834 | `"inference_cache": "off"` | Error handling test |
| `e2e/endpoints/internal/evaluations.rs` | `test_run_evaluation_streaming_nonexistent_dataset` | 873 | `"inference_cache": "off"` | Error handling test |
| `e2e/endpoints/internal/evaluations.rs` | `test_run_evaluation_streaming_with_specific_datapoint_ids` | 943 | `"inference_cache": "off"` | Datapoint filtering test |
| `e2e/endpoints/internal/evaluations.rs` | `test_run_evaluation_streaming_conflicting_variant_config` | 1027 | `"inference_cache": "off"` | Validation logic test |

### Category 4: Raw Usage Cache Tests with UUID

| File | Test | Line | Bypass Method | Purpose |
|------|------|------|---------------|---------|
| `e2e/raw_usage/cache.rs` | `test_raw_usage_cache_behavior_non_streaming` | 125 | `Uuid::now_v7()` in unique_input | Test first call is cache miss |
| `e2e/raw_usage/cache.rs` | `test_raw_usage_cache_behavior_streaming` | 154 | `Uuid::now_v7()` in unique_input | Streaming variant |

---

## Recommended Actions

### Phase 1: High-Priority Fixes (Remove Randomness for Deterministic Tests)

These tests inject randomness but don't actually need live provider responses - they just need deterministic, unique cache keys per test run.

#### 1.1 Cache Tests - Use Fixed Seeds

**File: `tensorzero-core/tests/e2e/cache.rs`**

For tests that need unique cache entries but not live responses:
- Replace `rand::random::<u32>()` with fixed test-specific seeds
- Replace `Uuid::now_v7()` with deterministic UUIDs or fixed strings
- Each test can use a unique but fixed string that ensures no cache collisions between tests

```rust
// Before:
let seed = rand::random::<u32>();

// After:
const TEST_SEED: u32 = 12345; // Or use test name hash
```

**Tests to fix:**
- `test_cache_write_and_read` (line 63)
- `test_cache_stream_write_and_read` (line 194)
- `test_streaming_cache_with_err` (line 478)
- `test_streaming_cache_without_err` (line 492)
- `test_dont_cache_invalid_tool_call` (line 368)
- `test_dont_cache_tool_call_schema_error` (line 418)

#### 1.2 Provider Tests - Use Fixed Identifiers

**File: `tensorzero-core/tests/e2e/providers/anthropic.rs`**

The test `test_thinking_rejected_128k` uses `Uuid::now_v7()` to bypass provider-proxy cache and verify current Anthropic behavior. This test should:
- Either use a fixed identifier and rely on cached response
- Or be moved to a periodic live-only test suite

**File: `tensorzero-core/tests/e2e/providers/common.rs`**

The test `test_bad_auth_extra_headers_with_provider_and_stream` uses `Uuid::now_v7()` for auth error testing. This should:
- Use a fixed identifier since error responses are deterministic
- Provider-proxy already doesn't cache error responses

### Phase 2: Evaluation Tests - Require Special Handling

The evaluation tests explicitly disable caching because evaluations need fresh model inferences. Options:

1. **Create deterministic evaluation fixtures**: Pre-populate cache with expected inference responses
2. **Mock the evaluation endpoint**: Use mock-provider-api for deterministic responses
3. **Move to periodic live tests**: Run these only in scheduled CI, not merge queue

Recommended: Option 2 (use mock-provider-api) for tests that primarily verify evaluation logic, not model behavior.

### Phase 3: Raw Usage Tests - Already Partially Correct

These tests use `Uuid::now_v7()` but the first call intentionally needs to be a cache miss to test the raw_usage behavior difference. However:

- `test_raw_usage_cache_behavior_*` - Could use fixed unique strings per test
- `test_raw_usage_cache_disabled*` - Need live responses; move to periodic tests or use mocks

### Phase 4: CI Pipeline Changes

1. **Merge Queue Configuration**:
   - Change provider-proxy mode from `ReadOldWriteNew` to `ReadOnly`
   - This ensures merge queue never hits live providers
   - Tests will fail if cache is missing (good - forces fixture regeneration)

2. **New Periodic Live Test Job**:
   - Create separate workflow that runs with `ReadWrite` mode
   - Schedule: Daily or on-demand
   - Does not block merge queue
   - Sends Slack alerts on failure instead of failing CI

3. **Cache Regeneration Workflow**:
   - Manual trigger to refresh provider-proxy cache
   - Runs all tests with live providers
   - Uploads new cache to R2

---

## Implementation Checklist

### Immediate (No Code Changes)

- [ ] Document which tests MUST use live providers vs. can use cached
- [ ] Inventory all tests with randomization patterns

### Code Changes Required

#### `tensorzero-core/tests/e2e/cache.rs`
- [ ] Replace `rand::random::<u32>()` with fixed seeds (4 tests)
- [ ] Replace `Uuid::now_v7()` with fixed strings (2 tests)

#### `tensorzero-core/tests/e2e/providers/anthropic.rs`
- [ ] Change `test_thinking_rejected_128k` to use fixed identifier
- [ ] Or mark as `#[ignore]` for merge queue, run in periodic tests

#### `tensorzero-core/tests/e2e/providers/common.rs`
- [ ] Change `test_bad_auth_extra_headers_with_provider_and_stream` to use fixed identifier

#### `tensorzero-core/tests/e2e/raw_usage/cache.rs`
- [ ] Replace `Uuid::now_v7()` with fixed unique strings (2 tests)
- [ ] Consider mocking for `cache_disabled` tests (4 tests)

#### `tensorzero-core/tests/e2e/endpoints/internal/evaluations.rs`
- [ ] Either mock provider responses or move to periodic tests (5 tests)

### CI Changes

#### `.github/workflows/merge-queue.yml`
- [ ] Add provider-proxy `--mode read-only` flag for merge queue runs
- [ ] Remove `ReadOldWriteNew` behavior for merge queue

#### New Workflow: `.github/workflows/live-provider-tests.yml`
- [ ] Create scheduled workflow for live provider testing
- [ ] Configure Slack notifications for failures
- [ ] Run with `--mode read-write` to refresh cache

---

## Summary

**Total tests requiring changes: ~20 tests across 5 files**

| Category | Test Count | Recommended Action |
|----------|------------|-------------------|
| Cache tests with random seeds | 6 | Use fixed seeds |
| Provider tests with UUIDs | 2 | Use fixed identifiers or move to periodic |
| Raw usage tests with UUIDs | 2 | Use fixed unique strings |
| Raw usage cache disabled | 4 | Use mocks or move to periodic |
| Evaluation tests | 5 | Use mocks or move to periodic |

**CI Changes:**
- Merge queue: Switch to `ReadOnly` cache mode
- New periodic live test workflow
- Cache regeneration workflow
