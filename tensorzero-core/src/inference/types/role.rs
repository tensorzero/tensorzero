//! The role of a TensorZero input message: user or assistant
//!
//! This module re-exports the Role type from tensorzero-types.

pub use tensorzero_types::{
    ASSISTANT_TEXT_TEMPLATE_VAR, Role, SYSTEM_TEXT_TEMPLATE_VAR, USER_TEXT_TEMPLATE_VAR,
};
