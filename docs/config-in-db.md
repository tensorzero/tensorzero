# Config in DB

Two modes, mutually exclusive: **file mode** (existing TOML, unchanged) or **DB mode** (new, for marketplace/hosted). Set via `TENSORZERO_CONFIG_SOURCE=file|db`.

## Storage

```sql
CREATE TABLE config (
    id INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    value JSONB NOT NULL,          -- full config blob, same shape as UninitializedConfig
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE variant_versions (
    id UUID PRIMARY KEY,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    config JSONB NOT NULL,         -- full variant config with inline templates
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE config_policies (
    role TEXT PRIMARY KEY,
    allowed_paths TEXT[] NOT NULL   -- glob patterns
);
```

**Mutable** (config blob): experimentation weights, models, metrics, tools, evaluators, optimizers — latest value wins, overwrite in place.

**Immutable** (variant_versions): prompt templates, model params — append-only, never overwritten. Config blob references versions by ID.

## Endpoints

| Endpoint | Verb | Purpose |
|---|---|---|
| `GET /config?path=...` | GET | Read full config or subtree |
| `PATCH /config` | PATCH | Update mutable config via JSON path |
| `DELETE /config?path=...` | DELETE | Remove a key from config |
| `POST /config/variants` | POST | Create immutable variant version (optionally activate) |
| `GET /config/variants?function=...&variant=...` | GET | List variant version history |

## Path → Type Mapping

The JSON path *is* the router. No per-path type registry — serde validates the whole blob after every mutation.

```
Path                                          Rust Type
────                                          ─────────
functions.{name}                              UninitializedFunctionConfig (Chat | Json)
functions.{name}.variants.{name}              VariantReference { version_id: Uuid }
functions.{name}.experimentation              UninitializedExperimentationConfig
functions.{name}.system_schema                String (inline JSON schema)
models.{name}                                 UninitializedModelConfig
models.{name}.providers.{name}                UninitializedModelProvider
metrics.{name}                                MetricConfig { type, optimize, level }
tools.{name}                                  UninitializedToolConfig
evaluations.{name}                            UninitializedEvaluationConfig
optimizers.{name}                             UninitializedOptimizerInfo
```

In DB mode, filesystem paths (templates, schemas) become inline strings. Same field name, content instead of path.

## Request Lifecycle

```
PATCH /config
Authorization: Bearer <token>
{ "path": "models.gpt4o.providers.azure", "value": { "config": { "type": "azure", ... } } }

1. AUTH      token → role:"operator"
2. RBAC      "models.gpt4o.providers.azure" vs role patterns → "models.**" matches → allowed
3. APPLY     jsonb_set(value, '{models,gpt4o,providers,azure}', '...')
4. VALIDATE  serde_json::from_value::<UninitializedConfig>(full_blob)
             → Ok:  COMMIT + touch updated_at
             → Err: ROLLBACK + return serde error with exact path
5. RESPOND   200 + updated subtree
```

Delete uses `value #- '{...}'`. Read uses `value #> '{...}'`. Same validation after every mutation.

### Bulk operations

Accept an array of patches, all-or-nothing:

```json
[
  { "path": "models.new_model", "value": { ... } },
  { "path": "functions.my_func.variants.v2", "value": { "version_id": "uuid" } }
]
```

Every path checked against RBAC, all applied, one validation pass, one commit.

## RBAC

### Glob patterns

`*` matches one level, `**` matches recursive. Stored in `config_policies`.

| Role | Patterns | Rationale |
|---|---|---|
| `autopilot` | `functions.*.variants.*`, `functions.*.experimentation.**` | Create/activate variants, change weights. Can't touch models, metrics, tools. |
| `operator` | `functions.**`, `models.**`, `metrics.**`, `tools.**`, `evaluations.**` | Full functional config. Can't touch optimizers or autopilot settings. |
| `admin` | `**` | Everything. |

### Variant-specific RBAC

Creating a variant version (`POST /config/variants`) is always allowed for authorized roles — it just inserts an immutable row. *Activating* it (patching `functions.{name}.variants.{name}` in the config blob) goes through normal path-based RBAC. This lets autopilot propose without activating, and operators review then activate.

### Policy storage

Policies live in `config_policies`, not in the mutable config blob — otherwise a compromised role could escalate its own permissions.

## Hot Reload

Gateway polls `config.updated_at` on a configurable interval (default 60s). On change: load blob → resolve variant references → deserialize → validate → atomic swap. Old config stays live if validation fails (error logged).

## What lives where

**Env vars / deploy-time config:** `gateway` (bind address, auth), `postgres` (connection string), `clickhouse`, `object_storage`, `rate_limiting`.

**DB config blob:** `functions`, `models`, `metrics`, `tools`, `evaluations`, `optimizers`, `provider_types`, `autopilot`.

**DB variant_versions:** Immutable variant snapshots (prompt templates, model parameters). Referenced by ID from config blob. Inference records store `variant_version_id` for historical replay.
