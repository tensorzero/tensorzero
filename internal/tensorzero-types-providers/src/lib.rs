//! Wire format types for external provider APIs.
//!
//! This crate contains serde types for communicating with external model provider APIs.

pub mod aws_bedrock;
pub mod deepseek;
pub mod groq;
pub mod mistral;
pub mod openai;
pub mod openrouter;
pub mod serde_util;
pub mod xai;
