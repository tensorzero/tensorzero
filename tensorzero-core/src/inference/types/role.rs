//! The role of a TensorZero input message: user or assistant

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use tensorzero_derive::export_schema;

use crate::variant::chat_completion::{ASSISTANT_TEXT_TEMPLATE_VAR, USER_TEXT_TEMPLATE_VAR};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS, JsonSchema)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "pyo3", pyclass)]
#[export_schema]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    /// The template name to use for `{"type": "text", "arguments": {}}` inputs.
    /// This will eventually be deprecated in favor of explicit `{"type": "template", "name": "user", "arguments": {}}` inputs.
    pub fn implicit_template_name(&self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }

    pub fn implicit_template_var(&self) -> &'static str {
        match self {
            Role::User => USER_TEXT_TEMPLATE_VAR,
            Role::Assistant => ASSISTANT_TEXT_TEMPLATE_VAR,
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
        }
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Role {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}
