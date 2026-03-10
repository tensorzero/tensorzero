//! GEPA optimizer implementation

use tensorzero_core::variant::chat_completion::UninitializedChatCompletionConfig;

use evaluate::VariantName;

pub mod analyze;
pub mod durable;
pub mod evaluate;
pub mod mutate;
pub mod pareto;
pub(crate) mod sequential;
pub mod validate;

/// A GEPA variant with its name and configuration
#[derive(Debug)]
pub struct GEPAVariant {
    pub name: VariantName,
    pub config: UninitializedChatCompletionConfig,
}
