//! Type definitions for OpenAI-compatible API.
//!
//! This module contains all the type definitions used by the OpenAI-compatible endpoints,
//! organized into submodules for chat completions, embeddings, input files, streaming,
//! tools, and usage tracking.

pub mod chat_completions;
pub mod embeddings;
pub mod input_files;
pub mod streaming;
pub mod tool;
pub mod usage;

/// Helper for serde `skip_serializing_if` - returns true if the Option is None or contains an empty Vec.
// Signature dictated by Serde
#[expect(clippy::ref_option)]
pub(crate) fn is_none_or_empty<T>(v: &Option<Vec<T>>) -> bool {
    v.as_ref().is_none_or(Vec::is_empty)
}
