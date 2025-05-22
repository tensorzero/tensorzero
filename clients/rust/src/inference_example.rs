use serde_json::Value;
use tensorzero_internal::{
    inference::types::{
        ContentBlockChatOutput, ContentBlockOutput, JsonInferenceOutput, ModelInput, ResolvedInput,
    },
    tool::ToolCallConfigDatabaseInsert,
};
use uuid::Uuid;

pub struct ChatInferenceExample {
    pub function_name: String,
    pub variant_name: String,
    pub input: ResolvedInput,
    pub output: Vec<ContentBlockChatOutput>,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    pub tool_params: ToolCallConfigDatabaseInsert,
}

pub struct JsonInferenceExample {
    pub function_name: String,
    pub variant_name: String,
    pub input: ResolvedInput,
    pub output: JsonInferenceOutput,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    pub output_schema: Value,
}

pub enum InferenceExample {
    Chat(ChatInferenceExample),
    Json(JsonInferenceExample),
}

pub struct RenderedStoredInference {
    function_name: String,
    variant_name: String,
    input: ModelInput,
    output: Vec<ContentBlockOutput>, // TODO(Viraj): check that this is the correct type
    episode_id: Uuid,
    inference_id: Uuid,
    tool_params: Option<ToolCallConfigDatabaseInsert>,
    output_schema: Option<Value>,
}
