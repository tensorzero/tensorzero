use tensorzero_core::endpoints::openai_compatible::*;
use utoipa::OpenApi;
use serde_json;

#[derive(OpenApi)]
#[openapi(
    paths(
        tensorzero_core::endpoints::openai_compatible::inference_handler,
        tensorzero_core::endpoints::openai_compatible::embeddings_handler,
    ),
    components(
        schemas(
            OpenAICompatibleEmbeddingParams,
            OpenAIEmbeddingResponse,
            OpenAIEmbedding,
            OpenAIEmbeddingUsage,
            OpenAICompatibleFunctionCall,
            OpenAICompatibleToolCallDelta,
            OpenAICompatibleToolCall,
            OpenAICompatibleToolCallChunk,
            OpenAICompatibleSystemMessage,
            OpenAICompatibleUserMessage,
            OpenAICompatibleAssistantMessage,
            OpenAICompatibleToolMessage,
            OpenAICompatibleMessage,
            OpenAICompatibleResponseFormat,
            JsonSchemaInfoOption,
            JsonSchemaInfo,
            OpenAICompatibleTool,
            FunctionName,
            OpenAICompatibleNamedToolChoice,
            ChatCompletionToolChoiceOption,
            OpenAICompatibleStreamOptions,
            OpenAICompatibleParams,
            OpenAICompatibleUsage,
            OpenAICompatibleResponseMessage,
            OpenAICompatibleChoice,
            OpenAICompatibleFinishReason,
            OpenAICompatibleResponse,
            OpenAICompatibleResponseChunk,
            OpenAICompatibleChoiceChunk,
            OpenAICompatibleDelta,
            OpenAICompatibleContentBlock,
            OpenAICompatibleImageUrl,
            OpenAICompatibleFile,
            TextContent
        )
    )
)]
struct ApiDoc;

#[tokio::test]
async fn test_generate_openapi_spec() {
    let spec = ApiDoc::openapi();
    let json_spec = serde_json::to_string_pretty(&spec).expect("Failed to serialize to JSON");
    
    println!("Generated OpenAPI spec:\n{}", json_spec);
    
    assert!(!json_spec.is_empty());
    assert!(json_spec.contains("\"openapi\""));
    
    println!(" OpenAPI spec generated successfully!");
}