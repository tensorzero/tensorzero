pub use tensorzero_provider_types::extra_body::*;

use tensorzero_stored_config::{
    StoredExtraBodyConfig, StoredExtraBodyReplacement, StoredExtraBodyReplacementKind,
};

// ─── Stored → Uninitialized conversions ──────────────────────────────────────
// Free functions instead of From impls to avoid orphan rule violations
// (both types are defined outside this crate).

pub fn extra_body_replacement_kind_from_stored(
    stored: StoredExtraBodyReplacementKind,
) -> ExtraBodyReplacementKind {
    match stored {
        StoredExtraBodyReplacementKind::Value(v) => ExtraBodyReplacementKind::Value(v),
        StoredExtraBodyReplacementKind::Delete => ExtraBodyReplacementKind::Delete,
    }
}

pub fn extra_body_replacement_from_stored(
    stored: StoredExtraBodyReplacement,
) -> ExtraBodyReplacement {
    ExtraBodyReplacement {
        pointer: stored.pointer,
        kind: extra_body_replacement_kind_from_stored(stored.kind),
    }
}

pub fn extra_body_config_from_stored(stored: StoredExtraBodyConfig) -> ExtraBodyConfig {
    ExtraBodyConfig {
        data: stored
            .data
            .into_iter()
            .map(extra_body_replacement_from_stored)
            .collect(),
    }
}
