//! Wire format types for external provider APIs.
//!
//! This crate contains serde types for communicating with external model provider APIs.
//!
//! See [`cache`] for a complete reference of which providers support prompt caching
//! and how their API fields map to TensorZero's internal `Usage` struct.

pub mod aws_bedrock;
pub mod cache;
pub mod conversions;
pub mod deepseek;
pub mod fireworks;
pub mod groq;
pub mod mistral;
pub mod openai;
pub mod openrouter;
pub mod serde_util;
pub mod together;
pub mod xai;
