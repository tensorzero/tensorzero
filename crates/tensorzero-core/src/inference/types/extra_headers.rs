pub use tensorzero_provider_types::extra_headers::*;

use tensorzero_stored_config::{
    StoredExtraHeader, StoredExtraHeaderKind, StoredExtraHeadersConfig,
};

// ─── Stored → Uninitialized conversions ──────────────────────────────────────
// Free functions instead of From impls to avoid orphan rule violations
// (both types are defined outside this crate).

pub fn extra_header_kind_from_stored(stored: StoredExtraHeaderKind) -> ExtraHeaderKind {
    match stored {
        StoredExtraHeaderKind::Value(v) => ExtraHeaderKind::Value(v),
        StoredExtraHeaderKind::Delete => ExtraHeaderKind::Delete,
    }
}

pub fn extra_header_from_stored(stored: StoredExtraHeader) -> ExtraHeader {
    ExtraHeader {
        name: stored.name,
        kind: extra_header_kind_from_stored(stored.kind),
    }
}

pub fn extra_headers_config_from_stored(stored: StoredExtraHeadersConfig) -> ExtraHeadersConfig {
    ExtraHeadersConfig {
        data: stored
            .data
            .into_iter()
            .map(extra_header_from_stored)
            .collect(),
    }
}
