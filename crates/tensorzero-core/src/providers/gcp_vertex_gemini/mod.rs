//! Compatibility shim — re-exports the moved provider plus the core-only `optimization` submodule.

pub use tensorzero_providers::gcp_vertex_gemini::*;

pub mod optimization;
