//! UUID to BIP39 triple substitution for LLM inference.
//!
//! # Overview
//!
//! This module provides bidirectional substitution between UUIDs and human-readable
//! BIP39 word triples (e.g., `apple-banana-cherry`). The primary purpose is to improve
//! LLM reliability when handling identifiers—LLMs often struggle with long hex UUIDs,
//! sometimes hallucinating or corrupting them, whereas short memorable word sequences
//! are easier for models to track and reproduce accurately.
//!
//! # How It Works
//!
//! 1. **Preprocessing**: Before sending messages to an LLM, all UUIDs in the input
//!    are replaced with deterministic BIP39 triples using [`preprocess_message`].
//!
//! 2. **Postprocessing**: After receiving the LLM response, any BIP39 triples that
//!    were previously registered are converted back to their original UUIDs using
//!    [`postprocess_response`].
//!
//! The mapping is deterministic—the same UUID always produces the same triple via
//! a blake3 hash of the UUID's lower 8 bytes (chosen for better entropy with `UUIDv7`).
//!
//! # BIP39 Triples
//!
//! [BIP39](https://github.com/bitcoin/bips/blob/master/bip-0039.mediawiki) is a standard
//! word list of 2048 common English words used in cryptocurrency seed phrases. We use
//! three words joined by hyphens (e.g., `abandon-ability-able`), providing 2048³ ≈ 8.6
//! billion possible combinations—enough to avoid collisions in practice while remaining
//! readable.
//!
//! # Limitations
//!
//! - Only triples that were registered during preprocessing are converted back. If the
//!   LLM invents a valid-looking BIP39 triple, it will not be converted.
//! - The substitution is per-session; a new `UuidSubstituter` starts with no mappings.

mod bip39;

#[cfg(test)]
mod tests;

use regex::Regex;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use tensorzero_core::inference::types::ContentBlockChatOutput;
use tensorzero_core::tool::ToolCallWrapper;
use thiserror::Error;
use uuid::Uuid;

use tensorzero_core::client::{InferenceResponse, InputMessage, InputMessageContent};

use self::bip39::BIP39_WORDS;

#[derive(Debug, Error)]
pub enum UuidSubstitutionError {
    #[error("UUID collision: {new_uuid} and {existing_uuid} both map to triple '{triple}'")]
    Collision {
        new_uuid: Uuid,
        existing_uuid: Uuid,
        triple: String,
    },
    #[error("regex capture group {group} missing for match")]
    MissingCaptureGroup { group: usize },
}

// UUID regex: 8-4-4-4-12 hex pattern
static UUID_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    #[expect(clippy::expect_used)]
    Regex::new(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
        .expect("UUID regex should be valid")
});

// BIP39 word set for validation
static BIP39_WORD_SET: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| BIP39_WORDS.iter().copied().collect());

// Triple pattern: word-word-word (validated against BIP39 later, case-insensitive)
static TRIPLE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    #[expect(clippy::expect_used)]
    let regex = Regex::new(r"(?i)\b([a-z]+)-([a-z]+)-([a-z]+)\b")
        .expect("triple BIP word regex should be valid");
    regex
});

#[derive(Debug, Clone, Default)]
pub struct UuidSubstituter {
    uuid_to_triple: HashMap<Uuid, String>,
    triple_to_uuid: HashMap<String, Uuid>,
}

impl UuidSubstituter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a triple for a UUID
    fn get_or_create_triple(&mut self, uuid: Uuid) -> Result<String, UuidSubstitutionError> {
        if let Some(triple) = self.uuid_to_triple.get(&uuid) {
            return Ok(triple.clone());
        }
        let triple = generate_triple(&uuid);
        if let Some(&existing_uuid) = self.triple_to_uuid.get(&triple) {
            return Err(UuidSubstitutionError::Collision {
                new_uuid: uuid,
                existing_uuid,
                triple,
            });
        }
        self.uuid_to_triple.insert(uuid, triple.clone());
        self.triple_to_uuid.insert(triple.clone(), uuid);
        Ok(triple)
    }

    /// Substitute UUIDs with triples in a string.
    ///
    /// # Errors
    ///
    /// Returns [`UuidSubstitutionError`] if two different UUIDs hash to the same BIP39 triple,
    /// or if a regex capture group is unexpectedly missing.
    pub fn substitute_uuids(&mut self, text: &str) -> Result<String, UuidSubstitutionError> {
        let mut error: Option<UuidSubstitutionError> = None;
        let result = UUID_REGEX
            .replace_all(text, |caps: &regex::Captures| {
                if error.is_some() {
                    return String::new();
                }
                let Some(uuid_match) = caps.get(0) else {
                    error = Some(UuidSubstitutionError::MissingCaptureGroup { group: 0 });
                    return String::new();
                };
                let uuid_str = uuid_match.as_str();
                match Uuid::parse_str(uuid_str) {
                    Ok(uuid) => match self.get_or_create_triple(uuid) {
                        Ok(triple) => triple,
                        Err(e) => {
                            error = Some(e);
                            String::new()
                        }
                    },
                    Err(_) => uuid_str.to_string(),
                }
            })
            .into_owned();
        match error {
            Some(e) => Err(e),
            None => Ok(result),
        }
    }

    /// Substitute triples back to UUIDs in a string
    ///
    /// # Errors
    ///
    /// Returns [`UuidSubstitutionError::MissingCaptureGroup`] if a regex capture group is unexpectedly missing.
    pub fn substitute_triples(&self, text: &str) -> Result<String, UuidSubstitutionError> {
        let mut error: Option<UuidSubstitutionError> = None;
        let result = TRIPLE_REGEX
            .replace_all(text, |caps: &regex::Captures| {
                if error.is_some() {
                    return String::new();
                }
                let Some(full_match) = caps.get(0) else {
                    error = Some(UuidSubstitutionError::MissingCaptureGroup { group: 0 });
                    return String::new();
                };
                let Some(w1_match) = caps.get(1) else {
                    error = Some(UuidSubstitutionError::MissingCaptureGroup { group: 1 });
                    return String::new();
                };
                let Some(w2_match) = caps.get(2) else {
                    error = Some(UuidSubstitutionError::MissingCaptureGroup { group: 2 });
                    return String::new();
                };
                let Some(w3_match) = caps.get(3) else {
                    error = Some(UuidSubstitutionError::MissingCaptureGroup { group: 3 });
                    return String::new();
                };
                let w1 = w1_match.as_str().to_lowercase();
                let w2 = w2_match.as_str().to_lowercase();
                let w3 = w3_match.as_str().to_lowercase();

                // Only substitute if all words are valid BIP39 and in our mapping
                if BIP39_WORD_SET.contains(w1.as_str())
                    && BIP39_WORD_SET.contains(w2.as_str())
                    && BIP39_WORD_SET.contains(w3.as_str())
                {
                    let lowercase_triple = format!("{w1}-{w2}-{w3}");
                    if let Some(uuid) = self.triple_to_uuid.get(&lowercase_triple) {
                        return uuid.to_string();
                    }
                }
                full_match.as_str().to_string()
            })
            .into_owned();
        match error {
            Some(e) => Err(e),
            None => Ok(result),
        }
    }

    /// Number of UUIDs registered
    pub fn len(&self) -> usize {
        self.uuid_to_triple.len()
    }

    /// Whether no UUIDs have been registered
    pub fn is_empty(&self) -> bool {
        self.uuid_to_triple.is_empty()
    }

    /// Inject a pre-existing UUID-to-triple mapping for testing collision detection.
    #[cfg(test)]
    fn inject_mapping(&mut self, uuid: Uuid, triple: String) {
        self.uuid_to_triple.insert(uuid, triple.clone());
        self.triple_to_uuid.insert(triple, uuid);
    }
}

pub fn preprocess_message(
    substituter: &mut UuidSubstituter,
    message: InputMessage,
) -> Result<InputMessage, UuidSubstitutionError> {
    let content = message
        .content
        .into_iter()
        .map(|content| preprocess_content_block(substituter, content))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(InputMessage {
        role: message.role,
        content,
    })
}

pub fn postprocess_response(
    substituter: &UuidSubstituter,
    response: InferenceResponse,
) -> Result<InferenceResponse, UuidSubstitutionError> {
    Ok(match response {
        InferenceResponse::Chat(mut chat) => {
            chat.content = chat
                .content
                .into_iter()
                .map(|content| postprocess_content_block(substituter, content))
                .collect::<Result<Vec<_>, _>>()?;
            InferenceResponse::Chat(chat)
        }
        InferenceResponse::Json(mut json) => {
            if let Some(ref mut raw) = json.output.raw {
                *raw = substituter.substitute_triples(raw)?;
            }
            if let Some(ref mut parsed) = json.output.parsed {
                process_text_in_value(parsed, &mut |s| substituter.substitute_triples(s))?;
            }
            InferenceResponse::Json(json)
        }
    })
}

fn preprocess_content_block(
    substituter: &mut UuidSubstituter,
    content: InputMessageContent,
) -> Result<InputMessageContent, UuidSubstitutionError> {
    Ok(match content {
        InputMessageContent::Text(mut text) => {
            text.text = substituter.substitute_uuids(&text.text)?;
            InputMessageContent::Text(text)
        }
        InputMessageContent::Template(mut template) => {
            for value in template.arguments.0.values_mut() {
                process_text_in_value(value, &mut |s| substituter.substitute_uuids(s))?;
            }
            InputMessageContent::Template(template)
        }
        InputMessageContent::ToolCall(wrapper) => {
            InputMessageContent::ToolCall(preprocess_tool_call_wrapper(substituter, wrapper)?)
        }
        InputMessageContent::ToolResult(mut result) => {
            result.result = substituter.substitute_uuids(&result.result)?;
            InputMessageContent::ToolResult(result)
        }
        InputMessageContent::RawText(mut raw) => {
            raw.value = substituter.substitute_uuids(&raw.value)?;
            InputMessageContent::RawText(raw)
        }
        InputMessageContent::Thought(mut thought) => {
            thought.text = thought
                .text
                .map(|text| substituter.substitute_uuids(&text))
                .transpose()?;
            InputMessageContent::Thought(thought)
        }
        InputMessageContent::File(file) => InputMessageContent::File(file),
        InputMessageContent::Unknown(unknown) => InputMessageContent::Unknown(unknown),
    })
}

fn preprocess_tool_call_wrapper(
    substituter: &mut UuidSubstituter,
    wrapper: ToolCallWrapper,
) -> Result<ToolCallWrapper, UuidSubstitutionError> {
    Ok(match wrapper {
        ToolCallWrapper::ToolCall(mut tool_call) => {
            tool_call.arguments = substituter.substitute_uuids(&tool_call.arguments)?;
            ToolCallWrapper::ToolCall(tool_call)
        }
        ToolCallWrapper::InferenceResponseToolCall(mut tool_call) => {
            tool_call.raw_arguments = substituter.substitute_uuids(&tool_call.raw_arguments)?;
            if let Some(arguments) = tool_call.arguments.as_mut() {
                process_text_in_value(arguments, &mut |s| substituter.substitute_uuids(s))?;
            }
            ToolCallWrapper::InferenceResponseToolCall(tool_call)
        }
    })
}

fn postprocess_content_block(
    substituter: &UuidSubstituter,
    content: ContentBlockChatOutput,
) -> Result<ContentBlockChatOutput, UuidSubstitutionError> {
    Ok(match content {
        ContentBlockChatOutput::Text(mut text) => {
            text.text = substituter.substitute_triples(&text.text)?;
            ContentBlockChatOutput::Text(text)
        }
        ContentBlockChatOutput::ToolCall(mut tool_call) => {
            tool_call.raw_arguments = substituter.substitute_triples(&tool_call.raw_arguments)?;
            if let Some(arguments) = tool_call.arguments.as_mut() {
                process_text_in_value(arguments, &mut |s| substituter.substitute_triples(s))?;
            }
            ContentBlockChatOutput::ToolCall(tool_call)
        }
        ContentBlockChatOutput::Thought(mut thought) => {
            thought.text = thought
                .text
                .map(|text| substituter.substitute_triples(&text))
                .transpose()?;
            ContentBlockChatOutput::Thought(thought)
        }
        ContentBlockChatOutput::Unknown(unknown) => ContentBlockChatOutput::Unknown(unknown),
    })
}

fn process_text_in_value<F>(value: &mut Value, f: &mut F) -> Result<(), UuidSubstitutionError>
where
    F: FnMut(&str) -> Result<String, UuidSubstitutionError>,
{
    match value {
        Value::String(text) => {
            *text = f(text.as_str())?;
        }
        Value::Array(values) => {
            for value in values {
                process_text_in_value(value, f)?;
            }
        }
        Value::Object(map) => {
            for value in map.values_mut() {
                process_text_in_value(value, f)?;
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
    Ok(())
}

fn generate_triple(uuid: &Uuid) -> String {
    let uuid_bytes = uuid.as_bytes();
    let hash = blake3::hash(uuid_bytes);
    let bytes = hash.as_bytes();

    // Extract 3 x 11-bit indices (BIP39 has 2048 = 2^11 words)
    let idx1 = (u16::from(bytes[0]) << 3) | (u16::from(bytes[1]) >> 5);
    let idx2 = ((u16::from(bytes[1]) & 0x1F) << 6) | (u16::from(bytes[2]) >> 2);
    let idx3 = ((u16::from(bytes[2]) & 0x03) << 9)
        | (u16::from(bytes[3]) << 1)
        | (u16::from(bytes[4]) >> 7);

    format!(
        "{}-{}-{}",
        BIP39_WORDS[usize::from(idx1 & 0x7FF)],
        BIP39_WORDS[usize::from(idx2 & 0x7FF)],
        BIP39_WORDS[usize::from(idx3 & 0x7FF)]
    )
}
