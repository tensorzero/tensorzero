#[cfg(feature = "integration_tests")]
mod tests {
    use api::inference::providers::anthropic;
    use api::inference::types::{
        FunctionType, InferenceRequestMessage, ModelInferenceRequest, Role,
    };
    use futures::StreamExt;
    use secrecy::SecretString;
    use std::env;

    #[tokio::test]
    async fn test_infer() {
        // Load API key from environment variable
        let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
        let api_key = SecretString::new(api_key);
        let model_name = "claude-3-haiku-20240307";
        let client = reqwest::Client::new();
        let messages = vec![
            InferenceRequestMessage {
                role: Role::System,
                content: "You are a helpful but mischevious assistant.".to_string(),
                tool_call_id: None,
            },
            InferenceRequestMessage {
                role: Role::User,
                content: "Is Santa Clause real?".to_string(),
                tool_call_id: None,
            },
        ];
        let max_tokens = Some(100);
        let temperature = Some(1.);
        let inference_request = ModelInferenceRequest {
            messages: messages.clone(),
            tools_available: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature,
            max_tokens,
            stream: false,
            json_mode: false,
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let result = anthropic::infer(inference_request, model_name, &client, &api_key).await;
        assert!(result.is_ok());
        assert!(result.unwrap().content.is_some());
    }

    #[tokio::test]
    async fn test_infer_stream() {
        // Load API key from environment variable
        let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
        let api_key = SecretString::new(api_key);
        let model_name = "claude-3-haiku-20240307";
        let client = reqwest::Client::new();
        let messages = vec![
            InferenceRequestMessage {
                role: Role::System,
                content: "You are a helpful but mischevious assistant.".to_string(),
                tool_call_id: None,
            },
            InferenceRequestMessage {
                role: Role::User,
                content: "Is Santa Clause real?".to_string(),
                tool_call_id: None,
            },
        ];
        let max_tokens = Some(100);
        let temperature = Some(1.);
        let inference_request = ModelInferenceRequest {
            messages: messages.clone(),
            tools_available: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature,
            max_tokens,
            stream: true,
            json_mode: false,
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let result =
            anthropic::infer_stream(inference_request, model_name, &client, &api_key).await;
        assert!(result.is_ok());
        let mut stream = result.unwrap();
        let mut collected_chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            assert!(chunk.is_ok());
            collected_chunks.push(chunk.unwrap());
        }
        assert!(!collected_chunks.is_empty());
        assert!(collected_chunks.last().unwrap().content.is_some());
    }
}
