use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tool::{ToolCall, ToolResult};

use super::{storage::StoragePath, Base64Image, Role, Thought};

/// Like `Input`, but with all network resources resolved.
/// Currently, this is just used to fetch image URLs in the image input,
/// so that we always pass a base64-encoded image to the model provider.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Value>,

    #[serde(default)]
    pub messages: Vec<ResolvedInputMessage>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedInputMessage {
    pub role: Role,
    pub content: Vec<ResolvedInputMessageContent>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResolvedInputMessageContent {
    Text {
        value: Value,
    },
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    RawText {
        value: String,
    },
    Thought(Thought),
    Image(ImageWithPath),
    Unknown {
        data: Value,
        model_provider_name: Option<String>,
    },
    // We may extend this in the future to include other types of content
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ImageWithPath {
    pub image: Base64Image,
    pub storage_path: StoragePath,
}
