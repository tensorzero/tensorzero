use serde_json::Value;
use tensorzero_internal::{
    config_parser::Config,
    inference::types::{
        ContentBlockChatOutput, ContentBlockOutput, JsonInferenceOutput, ModelInput, ResolvedInput,
    },
    tool::ToolCallConfigDatabaseInsert,
    variant::chat_completion::prepare_model_input,
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

impl InferenceExample {
    pub fn input_mut(&mut self) -> &mut ResolvedInput {
        match self {
            InferenceExample::Chat(example) => &mut example.input,
            InferenceExample::Json(example) => &mut example.input,
        }
    }
    pub fn input(&self) -> &ResolvedInput {
        match self {
            InferenceExample::Chat(example) => &example.input,
            InferenceExample::Json(example) => &example.input,
        }
    }
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

pub fn render_model_input(
    inference_example: &InferenceExample,
    config: &Config,
    variants: &HashMap<String, String>,
) -> Result<ModelInput, Error> {
    // TODO
    let variant_name = variants.get(&inference_example.function_name).ok_or_else(|| Error::new(ErrorDetails::InvalidVariant {
    let variant_config = config
        .get_function(&inference_example.function_name)?
        .variants()
        .get(variants)
        
        
        .unwrap();
    prepare_model_input(
        inference_example.input().system.as_ref(),
        &inference_example.input().messages,
        &config.templates,
        variants,
    );
}
