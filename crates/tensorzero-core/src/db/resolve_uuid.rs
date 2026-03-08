use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Error;

pub use tensorzero_types::{ResolveUuidResponse, ResolvedObject};

/// Trait for resolving a UUID to its object type(s).
#[async_trait]
pub trait ResolveUuidQueries {
    async fn resolve_uuid(&self, id: &Uuid) -> Result<Vec<ResolvedObject>, Error>;
}
