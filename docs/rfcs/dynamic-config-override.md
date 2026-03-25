# RFC: Dynamic Configuration Override via Database

**Status:** Draft
**Date:** 2026-03-19
**Authors:** TBD

## Problem Statement

TensorZero's configuration is currently defined entirely in TOML files, loaded once at gateway startup, and stored as an immutable `Arc<Config>` for the lifetime of the process. This creates friction for several important use cases:

1. **Autopilot-created variants**: After fine-tuning, autopilot writes a new config snapshot to the database via `POST /internal/config`, but the running gateway ignores it. The new variant only becomes active after a full gateway restart.

2. **Dynamic experimentation**: Changing variant weights, enabling/disabling variants, or rolling out a new variant requires editing TOML and restarting — unacceptable for production A/B testing.

3. **Programmatic variant creation**: External systems (optimizers, CI/CD pipelines, ML platforms) cannot create or update variants without filesystem access to the TOML config.

4. **Multi-tenant / platform use cases**: Platforms building on TensorZero need per-customer function/variant configuration without generating TOML files per tenant.

5. **Operational agility**: Kill-switching a broken variant, adjusting rate limits, or adding a new model provider should not require a deploy.

### What Exists Today

| Capability | Status | Notes |
|---|---|---|
| TOML config at startup | Done | Loaded into immutable `Arc<Config>` |
| Config snapshots in DB | Done | Blake3-hashed, stored in Postgres/ClickHouse |
| `POST /internal/config` | Done | Writes full config snapshot, validates before persisting |
| `GET /internal/config/{hash}` | Done | Loads historical config for reproducibility |
| Autopilot `write_config` tool | Done | Writes new full config snapshot via the internal API |
| Gateway hot-reload | **Missing** | Gateway never picks up new snapshots after startup |
| Granular config mutations | **Missing** | Only full-config writes supported; no PATCH semantics |
| Config diff / migration | **Missing** | No way to express "add variant X to function Y" without sending the entire config |

### Related GitHub Issues

- [#4713](https://github.com/tensorzero/tensorzero/issues/4713) — Dynamic tool configs in DB need clearer read/write semantics
- [#6874](https://github.com/tensorzero/tensorzero/issues/6874) — GET/POST config not symmetric (null serialization)
- [#6457](https://github.com/tensorzero/tensorzero/issues/6457) — UI stale config when autopilot creates entities (closed, workaround: retry-on-miss)
- [#6616](https://github.com/tensorzero/tensorzero/issues/6616) — RFC: Extract snapshot config resolution (closed)
- [#4844](https://github.com/tensorzero/tensorzero/issues/4844) — Rationalize file loading architecture across UI/evaluations/backend
- [#2551](https://github.com/tensorzero/tensorzero/issues/2551) — Dynamic variant config on inference endpoint (closed, implemented as `internal_dynamic_variant_config`)

---

## Design Goals

1. **Backwards compatible**: TOML-only deployments continue to work unchanged. Dynamic config is opt-in.
2. **Atomic and safe**: Config transitions are atomic — no partially-applied states. Invalid configs are rejected before they take effect.
3. **Auditable**: Every config state is a versioned snapshot with a hash. The full history is queryable.
4. **Granular**: Support targeted mutations (add variant, update weight, disable variant) without requiring a full config rewrite.
5. **Low latency**: Config changes should propagate to the gateway within seconds, not minutes.
6. **Reproducible**: Every inference continues to record the config hash it ran against. Historical configs remain loadable.

### Non-Goals (for v1)

- Real-time config push (WebSocket/SSE) — polling is sufficient.
- Per-request config override (already partially addressed by `internal_dynamic_variant_config`).
- Multi-gateway coordination / consensus (each gateway polls independently).
- Schema migrations of the config format itself.
- Tenant-level config isolation (a later extension).

---

## Proposed Architecture

### Overview

```
                    ┌──────────────────────┐
                    │   TOML Config Files   │  (base config, loaded at startup)
                    └──────────┬───────────┘
                               │
                               ▼
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Autopilot   │────▶│   Config Store   │◀────│  External APIs   │
│  Optimizers  │     │   (Postgres)     │     │  (PATCH /config) │
└──────────────┘     └────────┬────────┘     └──────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Config Poller     │  (every N seconds, checks for new hash)
                    │  (in gateway)      │
                    └─────────┬─────────┘
                              │ swap Arc<Config>
                    ┌─────────▼─────────┐
                    │  ArcSwap<Config>   │  (live config, atomically swappable)
                    │  in AppStateData   │
                    └───────────────────┘
```

### Component 1: Config Store (Postgres as source of truth)

The database becomes the **authoritative source** for the active config, not the filesystem. TOML files become the **initial seed** and the **developer ergonomic layer**.

**New table: `tensorzero.active_config`**

```sql
CREATE TABLE tensorzero.active_config (
    -- Singleton row (enforced by CHECK or unique key)
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    snapshot_hash BYTEA NOT NULL REFERENCES tensorzero.config_snapshots(hash),
    activated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    activated_by TEXT,  -- 'startup', 'autopilot:<session_id>', 'api:<user>'
    previous_hash BYTEA REFERENCES tensorzero.config_snapshots(hash)
);
```

**Behavior:**
- On gateway startup: load TOML → write snapshot → set as `active_config` if no active config exists (or if TOML has changed).
- On `POST /internal/config`: write snapshot → optionally activate it (new `activate: bool` field, default `false` for backwards compat).
- New endpoint `POST /internal/config/activate` with `{ "hash": "<hash>" }` to activate a previously written snapshot.

**Config activation log (for audit trail):**

```sql
CREATE TABLE tensorzero.config_activation_log (
    id BIGSERIAL PRIMARY KEY,
    snapshot_hash BYTEA NOT NULL,
    activated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    activated_by TEXT NOT NULL,
    previous_hash BYTEA,
    reason TEXT
);
```

### Component 2: Gateway Config Poller

A background task in the gateway that periodically checks `active_config` for changes.

```rust
// In gateway.rs, spawned during startup
async fn config_poll_loop(
    app_state: Arc<AppStateData>,
    poll_interval: Duration,  // default: 5s, configurable via gateway.config_poll_interval
    shutdown: CancellationToken,
) {
    let mut interval = tokio::time::interval(poll_interval);
    loop {
        tokio::select! {
            _ = interval.tick() => {
                match check_and_reload_config(&app_state).await {
                    Ok(Some(new_hash)) => {
                        tracing::info!(%new_hash, "Config reloaded");
                    }
                    Ok(None) => {} // No change
                    Err(e) => {
                        tracing::error!(error = %e, "Config reload failed, keeping current config");
                    }
                }
            }
            _ = shutdown.cancelled() => break,
        }
    }
}
```

**Reload logic:**
1. Query `active_config.snapshot_hash`
2. Compare against current `app_state.config.hash`
3. If different:
   a. Load snapshot from `config_snapshots` table
   b. Run full `Config::load_from_snapshot()` with credential validation
   c. Atomically swap `app_state.config` (requires `ArcSwap` — see below)
   d. Log the transition

### Component 3: ArcSwap for Live Config

**Current:** `config: Arc<Config>` in `AppStateData` — immutable after creation.

**Proposed:** Replace with `arc_swap::ArcSwap<Config>` for lock-free atomic swaps.

```rust
use arc_swap::ArcSwap;

#[derive(Clone)]
pub struct AppStateData {
    pub config: Arc<ArcSwap<Config>>,  // Changed from Arc<Config>
    // ... rest unchanged
}
```

**Access pattern change:**

```rust
// Before:
let config = &app_state.config;

// After:
let config = app_state.config.load();  // Returns Guard<Arc<Config>>, very cheap
```

`ArcSwap` is lock-free and wait-free for readers. The `load()` call is ~1 nanosecond overhead. Writers (the poller) call `store()` which is also lock-free. In-flight requests continue using their loaded `Arc<Config>` until they complete — no request sees a partial config state.

**Migration path:** This is a large but mechanical refactor. Every `app_state.config.X` becomes `app_state.config.load().X`. Can be done incrementally by:
1. First, alias `type ConfigRef = Arc<Config>` everywhere
2. Change `AppStateData.config` to `Arc<ArcSwap<Config>>`
3. Add a `config()` helper method that returns `arc_swap::Guard<Arc<Config>>`
4. Update call sites

### Component 4: Granular Config Mutation API

Instead of requiring a full config rewrite, provide targeted mutation endpoints.

#### Option A: JSON Patch (RFC 6902)

```
PATCH /internal/config
Content-Type: application/json-patch+json

[
  { "op": "add", "path": "/functions/my_chat/variants/v2", "value": { "type": "chat_completion", "model": "gpt-4o", ... } },
  { "op": "replace", "path": "/functions/my_chat/variants/v1/weight", "value": 0.0 },
  { "op": "remove", "path": "/functions/my_chat/variants/old_variant" }
]
```

**Pros:** Standard, expressive, supports any mutation.
**Cons:** Requires serializing config to JSON (we already do this), applying patches, then deserializing. Path-based addressing is fragile.

#### Option B: Domain-Specific Mutation Endpoints (Recommended for v1)

Higher-level endpoints that map to the operations users actually perform:

```
POST   /internal/config/functions/{name}/variants          — Create variant
PATCH  /internal/config/functions/{name}/variants/{name}   — Update variant (weight, model, templates, etc.)
DELETE /internal/config/functions/{name}/variants/{name}    — Remove variant
PATCH  /internal/config/functions/{name}/experimentation    — Update experimentation config (weights)

POST   /internal/config/models                              — Create model
PATCH  /internal/config/models/{name}                       — Update model (routing, providers)

POST   /internal/config/functions                           — Create function
PATCH  /internal/config/functions/{name}                    — Update function config

POST   /internal/config/tools                               — Create tool
PATCH  /internal/config/tools/{name}                        — Update tool

POST   /internal/config/activate                            — Activate a snapshot by hash
POST   /internal/config/rollback                            — Rollback to previous active config
```

**Mutation flow:**
1. Load current active config snapshot
2. Deserialize to `UninitializedConfig`
3. Apply the requested mutation
4. Re-serialize and create new `ConfigSnapshot`
5. Validate via `Config::load_from_snapshot()`
6. Write snapshot to DB
7. Set as active config
8. Gateway poller picks it up on next tick

**Pros:** Type-safe, self-documenting, easy to validate, natural audit trail.
**Cons:** More endpoints to implement and maintain.

#### Recommended: Start with Option B for the most common operations, add Option A later for power users.

**Priority order for mutation endpoints:**
1. Variant CRUD (highest demand — autopilot, optimizers, A/B testing)
2. Variant weight updates (experimentation)
3. Model CRUD (adding new providers/models)
4. Function CRUD
5. Tool CRUD

---

## Detailed Design: Variant CRUD (Priority 1)

### Create Variant

```
POST /internal/config/functions/{function_name}/variants
```

**Request:**
```json
{
  "name": "gpt4o_finetuned_v2",
  "type": "chat_completion",
  "model": "gpt-4o-ft-abc123",
  "weight": 0.0,
  "system_template": "You are a helpful assistant...",
  "user_template": "{{user_message}}",
  "temperature": 0.7,
  "max_tokens": 1024
}
```

**Response:**
```json
{
  "snapshot_hash": "abc123...",
  "variant_name": "gpt4o_finetuned_v2",
  "activated": true
}
```

**Validation:**
- Function must exist in current config
- Variant name must not already exist
- Model must exist in current config
- Templates must be valid MiniJinja
- Full config validation after mutation

### Update Variant Weight

```
PATCH /internal/config/functions/{function_name}/variants/{variant_name}
```

**Request:**
```json
{
  "weight": 0.5
}
```

This is the most common operation — ramping up a new variant after validation.

### Disable Variant (Kill Switch)

```
PATCH /internal/config/functions/{function_name}/variants/{variant_name}
```

**Request:**
```json
{
  "weight": 0.0
}
```

Setting weight to 0 effectively disables routing to the variant without removing it.

---

## Interaction with Existing Systems

### Autopilot

Today autopilot uses `write_config` to write a full config snapshot. With dynamic config:

1. **Short term**: Autopilot continues using `write_config` but adds `activate: true` to the request. The gateway poller picks it up.
2. **Medium term**: Autopilot uses granular mutation endpoints (`POST .../variants`) instead of rewriting the entire config. This is simpler, less error-prone, and produces better audit trails.

### Config Snapshots & Reproducibility

Every mutation creates a new snapshot with a new hash. The snapshot chain provides a complete history:

```
startup (hash: aaa) → add variant v2 (hash: bbb) → ramp v2 to 50% (hash: ccc) → rollback (hash: bbb)
```

Each inference still records `snapshot_hash`, so reproducibility is preserved.

### TOML Files

TOML files remain the **base config** and the developer-friendly authoring format:
- Used for initial seeding on first deploy
- Used for version-controlled "known good" configs
- Can be re-applied via `POST /internal/config` to reset to a known state
- **Not the source of truth after dynamic overrides are applied**

A new gateway config option controls behavior:

```toml
[gateway]
# "toml" (default, current behavior): TOML is authoritative, loaded every restart
# "database": DB is authoritative, TOML is only used for initial seeding
config_source = "database"

# How often to poll for config changes (only when config_source = "database")
config_poll_interval_seconds = 5
```

When `config_source = "database"`:
- On startup, check if `active_config` has a row
- If yes, load that snapshot (ignore TOML changes)
- If no, load TOML, write snapshot, set as active
- Start the config poller

When `config_source = "toml"` (default):
- Current behavior, no poller, no dynamic updates

### UI

The UI already polls for config changes (5s interval in `ui/app/utils/config/index.server.ts`). With gateway hot-reload:
- UI sees new entities immediately after the gateway reloads (within poll interval)
- Issue #6457 (stale config) becomes less severe — both gateway and UI are eventually consistent

### Evaluations

Evaluations already support historical config snapshots. No changes needed — they load the config snapshot associated with the inference being evaluated.

---

## Migration & Rollout Plan

### Phase 1: ArcSwap Foundation (Non-breaking)

1. Add `arc-swap` dependency
2. Replace `Arc<Config>` with `Arc<ArcSwap<Config>>` in `AppStateData`
3. Add `config()` helper method
4. Update all call sites (mechanical refactor)
5. **No behavior change** — config is still loaded once at startup

### Phase 2: Config Poller + Activation (Feature-flagged)

1. Add `active_config` and `config_activation_log` tables
2. Implement `POST /internal/config/activate` endpoint
3. Implement config poller (behind `config_source = "database"`)
4. Add `activate: bool` field to `WriteConfigRequest`
5. **Autopilot can now write + activate configs that take effect without restart**

### Phase 3: Granular Mutation API

1. Implement variant CRUD endpoints
2. Implement weight update endpoint
3. Update autopilot to use granular endpoints
4. Add remaining entity CRUD (models, functions, tools)

### Phase 4: Advanced Features

1. Config diff API (`GET /internal/config/diff/{hash1}/{hash2}`)
2. Rollback endpoint
3. Config validation dry-run (`POST /internal/config/validate`)
4. Webhook/callback on config change
5. JSON Patch support (Option A)

---

## Open Questions

1. **Conflict resolution**: If two writers (e.g., autopilot + human) mutate config simultaneously, last-write-wins via the `active_config` singleton. Is this sufficient, or do we need optimistic concurrency (e.g., `If-Match: <hash>` header)?

2. **TOML drift**: When `config_source = "database"`, the TOML files on disk will drift from the active config. Should we provide a `GET /internal/config/export` that returns the active config as TOML for version control?

3. **Credential handling**: Dynamic model creation needs credentials. Should credentials be stored in the database (encrypted), referenced by name from environment variables, or always require a gateway restart?

4. **Template storage**: Templates today are loaded from the filesystem and compiled into MiniJinja. Dynamic variants need templates stored in the database (already supported via `extra_templates` in snapshots). Should we formalize this as the primary template storage?

5. **Rate of change**: Should we rate-limit config mutations to prevent thrashing? (e.g., max 1 activation per second)

6. **Multi-gateway consistency**: In a multi-replica deployment, each gateway polls independently. There's a window where different replicas serve different configs. Is eventual consistency (bounded by poll interval) acceptable?

7. **Startup behavior with `config_source = "database"`**: If the database is unreachable at startup, should the gateway fall back to TOML, fail to start, or start with a degraded mode?

---

## Alternatives Considered

### 1. File-watching (inotify/fsevents)

Watch TOML files for changes and reload on modification.

**Rejected because:**
- Doesn't solve programmatic creation (still need filesystem access)
- Doesn't work in containerized/immutable deployments
- Platform-specific, unreliable with NFS/remote filesystems
- No audit trail

### 2. Full config replacement only (no granular mutations)

Only support writing complete config snapshots, no PATCH endpoints.

**Partially accepted:** This is the Phase 2 approach. Granular mutations in Phase 3 are a convenience layer on top.

### 3. Sidecar / external config service

Separate service that manages config and pushes to gateways.

**Rejected because:**
- Adds operational complexity
- TensorZero already has Postgres — no need for another service
- Config is tightly coupled to the gateway's validation logic

### 4. Environment variable overrides

Allow env vars to override specific config values (e.g., `TZ_FUNCTION_MY_CHAT_VARIANT_V1_WEIGHT=0.5`).

**Rejected because:**
- Doesn't scale beyond simple overrides
- No audit trail
- Requires process restart to pick up env changes
- Combinatorial explosion of override keys

---

## Appendix: Current Config Data Flow

```
TOML files on disk
    │
    ▼ (glob + merge)
toml::Table
    │
    ▼ (deserialize)
UninitializedConfig
    │
    ▼ (async initialization: load templates, validate models, build schemas)
Config + ConfigSnapshot
    │
    ▼ (write snapshot to DB)
Config (wrapped in Arc, placed in AppStateData)
    │
    ▼ (cloned via Arc for each request)
API Handlers (immutable access)
```

**Proposed data flow (with dynamic config):**

```
TOML files on disk (seed only when config_source = "database")
    │
    ▼ (on first deploy or reset)
ConfigSnapshot in Postgres (active_config pointer)
    │
    ▼ (poller detects new hash)
Config::load_from_snapshot()
    │
    ▼ (validate + initialize)
Config (swapped into ArcSwap)
    │
    ▼ (load() for each request — lock-free)
API Handlers (immutable access to current snapshot)
```
