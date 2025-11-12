use crate::providers::common::E2ETestProvider;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use tensorzero::test_helpers::make_embedded_gateway_with_config;
use tensorzero::{
    CacheParamsOptions, ClientInferenceParams, ClientInput, ClientInputMessage,
    ClientInputMessageContent, InferenceOutput, InferenceResponse,
};
use tensorzero_core::{
    cache::CacheEnabledMode,
    db::clickhouse::test_helpers::{get_clickhouse, select_model_inference_clickhouse},
    inference::types::{
        file::Base64File,
        storage::{StorageKind, StoragePath},
        ContentBlockChatOutput, File, Role, TextKind,
    },
};
use uuid::Uuid;

const INPUT_AUDIO_FUNCTION_CONFIG: &str = r#"
[functions.input_audio_test]
type = "chat"

[functions.input_audio_test.variants.azure]
type = "chat_completion"
model = "azure-gpt-4o-audio-preview"

[functions.input_audio_test.variants.gcp_vertex_gemini]
type = "chat_completion"
model = "gcp_vertex_gemini::projects/tensorzero-public/locations/us-central1/publishers/google/models/gemini-2.5-flash-lite"

[functions.input_audio_test.variants.google_ai_studio_gemini]
type = "chat_completion"
model = "google_ai_studio_gemini::gemini-2.5-flash-lite"

[functions.input_audio_test.variants.openai]
type = "chat_completion"
model = "openai::gpt-4o-audio-preview"

[functions.input_audio_test.variants.openrouter]
type = "chat_completion"
model = "openrouter::openai/gpt-4o-audio-preview"

[models.azure-gpt-4o-audio-preview]
routing = ["azure"]

[models.azure-gpt-4o-audio-preview.providers.azure]
type = "azure"
api_key_location = "env::AZURE_OPENAI_EASTUS2_API_KEY"
deployment_id = "gpt-4o-audio-preview"
endpoint = "https://t0-eastus2-resource.openai.azure.com"
"#;

/// Audio file of dogs barking
static AUDIO_FILE: &[u8] = include_bytes!("input_audio_barks.mp3");

/// BLAKE3 hash of the audio file for object storage
const AUDIO_FILE_HASH: &str = "4e497dd5ba1f3761a3d8bdf21da18632d4b919e66cba20af3bb1d07301fc7192";

pub async fn test_audio_inference_with_provider_filesystem(provider: E2ETestProvider) {
    let temp_dir = tempfile::tempdir().unwrap();
    let (_client, _storage_path) = test_base64_audio_inference_with_provider_and_store(
        provider,
        &StorageKind::Filesystem {
            path: temp_dir.path().to_string_lossy().to_string(),
        },
        &format!(
            r#"
        [object_storage]
        type = "filesystem"
        path = "{}"

        {INPUT_AUDIO_FUNCTION_CONFIG}
        "#,
            temp_dir.path().to_string_lossy()
        ),
        "",
    )
    .await;

    // Check that audio was stored in filesystem
    let result = std::fs::read(
        temp_dir
            .path()
            .join(format!("observability/files/{AUDIO_FILE_HASH}.mp3")),
    )
    .unwrap();
    assert!(
        result == AUDIO_FILE,
        "Audio in object store does not match expected audio"
    );
}

pub async fn test_base64_audio_inference_with_provider_and_store(
    provider: E2ETestProvider,
    kind: &StorageKind,
    config_toml: &str,
    prefix: &str,
) -> (tensorzero::Client, StoragePath) {
    let episode_id = Uuid::now_v7();

    let audio_data = BASE64_STANDARD.encode(AUDIO_FILE);

    let client = make_embedded_gateway_with_config(config_toml).await;
    let mut storage_path = None;

    for should_be_cached in [false, true] {
        let response = client
            .inference(ClientInferenceParams {
                function_name: Some("input_audio_test".to_string()),
                variant_name: Some(provider.variant_name.clone()),
                episode_id: Some(episode_id),
                input: ClientInput {
                    system: None,
                    messages: vec![ClientInputMessage {
                        role: Role::User,
                        content: vec![
                            ClientInputMessageContent::Text(TextKind::Text {
                                text: "What's going on in this audio?".to_string(),
                            }),
                            ClientInputMessageContent::File(File::Base64(
                                Base64File::new(
                                    None,
                                    "audio/mpeg".parse().unwrap(),
                                    audio_data.clone(),
                                    None,
                                    None,
                                )
                                .expect("test data should be valid"),
                            )),
                        ],
                    }],
                },
                cache_options: CacheParamsOptions {
                    enabled: CacheEnabledMode::On,
                    max_age_s: Some(10),
                },
                ..Default::default()
            })
            .await
            .unwrap();

        let InferenceOutput::NonStreaming(response) = response else {
            panic!("Expected non-streaming inference response");
        };

        let latest_storage_path = check_base64_audio_response(
            response,
            Some(episode_id),
            &provider,
            should_be_cached,
            kind,
            prefix,
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        storage_path = Some(latest_storage_path);
    }

    (client, storage_path.unwrap())
}

pub async fn check_base64_audio_response(
    response: InferenceResponse,
    episode_id: Option<Uuid>,
    provider: &E2ETestProvider,
    should_be_cached: bool,
    kind: &StorageKind,
    prefix: &str,
) -> StoragePath {
    // Extract content and inference_id based on response type
    let (content, response_episode_id, variant_name, inference_id) = match &response {
        InferenceResponse::Chat(chat) => (
            &chat.content,
            chat.episode_id,
            &chat.variant_name,
            chat.inference_id,
        ),
        InferenceResponse::Json(_) => panic!("Expected chat inference response"),
    };

    // Basic response checks
    assert_eq!(content.len(), 1);
    let text = match &content[0] {
        ContentBlockChatOutput::Text(text_block) => &text_block.text,
        _ => panic!("Expected text content block"),
    };
    assert!(!text.is_empty());

    // Check that the transcript contains "bark" or "dog"
    let transcript = text.to_lowercase();
    assert!(
        transcript.contains("bark") || transcript.contains("dog"),
        "Transcript should contain 'bark' or 'dog' but got: {text}"
    );

    // Check episode ID
    if let Some(expected_episode_id) = episode_id {
        assert_eq!(response_episode_id, expected_episode_id);
    }

    // Check variant name
    assert_eq!(variant_name, &provider.variant_name);

    // Check cache status in ModelInference table
    let clickhouse = get_clickhouse().await;
    let model_inference = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    assert_eq!(
        model_inference.get("cached").unwrap().as_bool().unwrap(),
        should_be_cached,
        "Expected cached={} but got cached={} for inference_id={}",
        should_be_cached,
        model_inference.get("cached").unwrap(),
        inference_id
    );

    // Return the storage path for the audio file
    object_store::path::Path::parse(format!("{prefix}observability/files/{AUDIO_FILE_HASH}.mp3"))
        .map(|path| StoragePath {
            kind: kind.clone(),
            path,
        })
        .unwrap()
}
