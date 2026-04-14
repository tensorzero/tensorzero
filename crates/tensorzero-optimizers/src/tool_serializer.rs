//! Serializers that project an owned `Vec<FunctionToolDef>` as a sequence of
//! provider-specific tool views at serialize time. This avoids the lifetime
//! gymnastics that would otherwise be required to store borrowed
//! `OpenAITool<'_>` / `OpenAISFTTool<'_>` / `FireworksTool<'_>` alongside the
//! `FunctionToolDef`s they borrow from in the same supervised-row struct.

use serde::Serializer;
use serde::ser::SerializeSeq;
use tensorzero_inference_types::FunctionToolDef;
use tensorzero_providers::fireworks::FireworksTool;
use tensorzero_providers::openai::{OpenAISFTTool, OpenAITool};

pub fn serialize_as_openai_sft_tools<S: Serializer>(
    defs: &[FunctionToolDef],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut seq = serializer.serialize_seq(Some(defs.len()))?;
    for def in defs {
        seq.serialize_element(&OpenAISFTTool::from(def))?;
    }
    seq.end()
}

pub fn serialize_as_openai_tools<S: Serializer>(
    defs: &[FunctionToolDef],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut seq = serializer.serialize_seq(Some(defs.len()))?;
    for def in defs {
        seq.serialize_element(&OpenAITool::from(def))?;
    }
    seq.end()
}

pub fn serialize_as_fireworks_tools<S: Serializer>(
    defs: &[FunctionToolDef],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut seq = serializer.serialize_seq(Some(defs.len()))?;
    for def in defs {
        seq.serialize_element(&FireworksTool::from(def))?;
    }
    seq.end()
}
