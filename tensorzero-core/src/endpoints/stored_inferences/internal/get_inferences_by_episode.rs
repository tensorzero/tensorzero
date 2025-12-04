use axum::extract::{Path, Query, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use crate::config::Config;
use crate::db::inferences::{InferenceOutputSource, InferenceQueries, ListInferencesParams};
use crate::error::Error;
use crate::stored_inference::{StoredInference, StoredInferenceDatabase};
use crate::utils::gateway::{AppState, AppStateData};

const DEFAULT_LIMIT: u32 = 100;

fn default_limit() -> u32 {
    DEFAULT_LIMIT
}

#[derive(Debug, Deserialize)]
pub struct GetInferencesByEpisodeParams {
    pub function_name: Option<String>,
    pub variant_name: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: u32,
    #[serde(default)]
    pub offset: u32,
    #[serde(default)]
    pub deduplicate: bool,
}

#[derive(Debug, Deserialize)]
pub struct EpisodePathParams {
    pub episode_id: Uuid,
}

#[derive(Debug, Serialize)]
pub struct GetInferencesByEpisodeResponse {
    pub inferences: Vec<StoredInference>,
}

/// Handler for the GET `/internal/episode/{episode_id}/inferences` endpoint.
/// Retrieves inferences for a specific episode with optional filtering and deduplication.
#[axum::debug_handler(state = AppStateData)]
#[instrument(
    name = "internal.get_inferences_by_episode",
    skip(app_state, query_params)
)]
pub async fn get_inferences_by_episode_handler(
    State(app_state): AppState,
    Path(path_params): Path<EpisodePathParams>,
    Query(query_params): Query<GetInferencesByEpisodeParams>,
) -> Result<Json<GetInferencesByEpisodeResponse>, Error> {
    let response = get_inferences_by_episode(
        &app_state.config,
        &app_state.clickhouse_connection_info,
        path_params.episode_id,
        query_params,
    )
    .await?;
    Ok(Json(response))
}

pub async fn get_inferences_by_episode(
    config: &Config,
    clickhouse: &impl InferenceQueries,
    episode_id: Uuid,
    params: GetInferencesByEpisodeParams,
) -> Result<GetInferencesByEpisodeResponse, Error> {
    let list_params = ListInferencesParams {
        episode_id: Some(&episode_id),
        function_name: params.function_name.as_deref(),
        variant_name: params.variant_name.as_deref(),
        output_source: InferenceOutputSource::Inference,
        limit: params.limit,
        offset: params.offset,
        ..Default::default()
    };

    let inferences_storage = clickhouse.list_inferences(config, &list_params).await?;

    let mut inferences: Vec<StoredInference> = inferences_storage
        .into_iter()
        .map(StoredInferenceDatabase::into_stored_inference)
        .collect::<Result<Vec<_>, _>>()?;

    // Apply deduplication if requested
    if params.deduplicate {
        inferences = deduplicate_inferences(inferences);
    }

    Ok(GetInferencesByEpisodeResponse { inferences })
}

/// Deduplicates inferences by removing those whose inputs are prefixes of other inferences.
///
/// Algorithm:
/// 1. Sort inferences by input message count DESC (most messages first)
/// 2. For each inference, check if it's a prefix of any already-accepted leaf
/// 3. If NOT a prefix of any leaf, add it to the result
/// 4. Re-sort by timestamp ASC before returning
///
/// Two inferences are considered for prefix comparison only if:
/// - They have the same function_name
/// - They have the same system prompt (input.system)
/// - The messages of one are a prefix of the messages of the other
fn deduplicate_inferences(mut inferences: Vec<StoredInference>) -> Vec<StoredInference> {
    if inferences.is_empty() {
        return inferences;
    }

    // Sort by message count DESC (most messages first)
    inferences.sort_by(|a, b| {
        let a_msg_count = get_message_count(a);
        let b_msg_count = get_message_count(b);
        b_msg_count.cmp(&a_msg_count)
    });

    let mut leaves: Vec<StoredInference> = Vec::new();

    for inference in inferences {
        let is_prefix_of_any_leaf = leaves.iter().any(|leaf| is_prefix_of(&inference, leaf));

        if !is_prefix_of_any_leaf {
            leaves.push(inference);
        }
    }

    // Re-sort by timestamp ASC
    leaves.sort_by_key(get_timestamp);

    leaves
}

/// Check if `a` is a prefix of `b`.
/// Returns true if:
/// 1. function_name is identical
/// 2. input.system is identical
/// 3. a's messages are a prefix of b's messages
fn is_prefix_of(a: &StoredInference, b: &StoredInference) -> bool {
    // Check function name
    if get_function_name(a) != get_function_name(b) {
        return false;
    }

    // Check system prompt equality
    if get_system(a) != get_system(b) {
        return false;
    }

    // Check if a's messages are a prefix of b's messages
    let a_messages = get_messages(a);
    let b_messages = get_messages(b);

    if a_messages.len() > b_messages.len() {
        return false;
    }

    // Compare each message - a's messages must match the first n messages of b
    a_messages
        .iter()
        .zip(b_messages.iter())
        .all(|(a_msg, b_msg)| a_msg == b_msg)
}

fn get_message_count(inference: &StoredInference) -> usize {
    match inference {
        StoredInference::Chat(chat) => chat.input.messages.len(),
        StoredInference::Json(json) => json.input.messages.len(),
    }
}

fn get_function_name(inference: &StoredInference) -> &str {
    match inference {
        StoredInference::Chat(chat) => &chat.function_name,
        StoredInference::Json(json) => &json.function_name,
    }
}

fn get_system(inference: &StoredInference) -> Option<&crate::inference::types::System> {
    match inference {
        StoredInference::Chat(chat) => chat.input.system.as_ref(),
        StoredInference::Json(json) => json.input.system.as_ref(),
    }
}

fn get_messages(
    inference: &StoredInference,
) -> &[crate::inference::types::stored_input::StoredInputMessage] {
    match inference {
        StoredInference::Chat(chat) => &chat.input.messages,
        StoredInference::Json(json) => &json.input.messages,
    }
}

fn get_timestamp(inference: &StoredInference) -> chrono::DateTime<chrono::Utc> {
    match inference {
        StoredInference::Chat(chat) => chat.timestamp,
        StoredInference::Json(json) => json.timestamp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, SchemaData};
    use crate::db::inferences::MockInferenceQueries;
    use crate::experimentation::ExperimentationConfig;
    use crate::function::{FunctionConfig, FunctionConfigChat};
    use crate::inference::types::stored_input::{StoredInput, StoredInputMessage};
    use crate::inference::types::{ContentBlockChatOutput, Role, StoredInputMessageContent, Text};
    use crate::stored_inference::{StoredChatInference, StoredChatInferenceDatabase};
    use crate::tool::{DynamicToolParams, ToolCallConfigDatabaseInsert, ToolChoice};
    use chrono::Utc;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn create_test_config() -> Config {
        let mut config = Config::default();
        config.functions.insert(
            "test_function".to_string(),
            Arc::new(FunctionConfig::Chat(FunctionConfigChat {
                variants: Default::default(),
                schemas: SchemaData::default(),
                tools: vec![],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: None,
                description: None,
                experimentation: ExperimentationConfig::default(),
                all_explicit_templates_names: Default::default(),
            })),
        );
        config
    }

    fn create_test_message(text: &str) -> StoredInputMessage {
        StoredInputMessage {
            role: Role::User,
            content: vec![StoredInputMessageContent::Text(Text {
                text: text.to_string(),
            })],
        }
    }

    fn create_test_inference(
        id: Uuid,
        function_name: &str,
        system: Option<&str>,
        messages: Vec<&str>,
        timestamp: chrono::DateTime<Utc>,
    ) -> StoredInference {
        StoredInference::Chat(StoredChatInference {
            function_name: function_name.to_string(),
            variant_name: "test_variant".to_string(),
            input: StoredInput {
                system: system.map(|s| crate::inference::types::System::Text(s.to_string())),
                messages: messages.iter().map(|m| create_test_message(m)).collect(),
            },
            output: vec![ContentBlockChatOutput::Text(Text {
                text: "test output".to_string(),
            })],
            dispreferred_outputs: vec![],
            timestamp,
            episode_id: Uuid::now_v7(),
            inference_id: id,
            tool_params: DynamicToolParams::default(),
            tags: HashMap::new(),
        })
    }

    fn create_test_inference_database(id: Uuid, episode_id: Uuid) -> StoredInferenceDatabase {
        StoredInferenceDatabase::Chat(StoredChatInferenceDatabase {
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            input: StoredInput {
                system: None,
                messages: vec![create_test_message("hello")],
            },
            output: vec![ContentBlockChatOutput::Text(Text {
                text: "test output".to_string(),
            })],
            dispreferred_outputs: vec![],
            timestamp: Utc::now(),
            episode_id,
            inference_id: id,
            tool_params: ToolCallConfigDatabaseInsert::default(),
            tags: HashMap::new(),
        })
    }

    #[tokio::test]
    async fn test_get_inferences_by_episode_basic() {
        let config = create_test_config();
        let episode_id = Uuid::now_v7();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();

        let inference1 = create_test_inference_database(id1, episode_id);
        let inference2 = create_test_inference_database(id2, episode_id);

        let mut mock_clickhouse = MockInferenceQueries::new();
        mock_clickhouse
            .expect_list_inferences()
            .withf(move |_, params| {
                assert_eq!(params.episode_id, Some(&episode_id));
                assert_eq!(params.limit, DEFAULT_LIMIT);
                assert_eq!(params.offset, 0);
                true
            })
            .times(1)
            .returning(move |_, _| {
                let inf1 = inference1.clone();
                let inf2 = inference2.clone();
                Box::pin(async move { Ok(vec![inf1, inf2]) })
            });

        let params = GetInferencesByEpisodeParams {
            function_name: None,
            variant_name: None,
            limit: DEFAULT_LIMIT,
            offset: 0,
            deduplicate: false,
        };

        let result = get_inferences_by_episode(&config, &mock_clickhouse, episode_id, params)
            .await
            .unwrap();

        assert_eq!(result.inferences.len(), 2);
    }

    #[tokio::test]
    async fn test_get_inferences_by_episode_with_filters() {
        let config = create_test_config();
        let episode_id = Uuid::now_v7();

        let mut mock_clickhouse = MockInferenceQueries::new();
        mock_clickhouse
            .expect_list_inferences()
            .withf(move |_, params| {
                assert_eq!(params.episode_id, Some(&episode_id));
                assert_eq!(params.function_name, Some("test_function"));
                assert_eq!(params.variant_name, Some("test_variant"));
                assert_eq!(params.limit, 50);
                assert_eq!(params.offset, 10);
                true
            })
            .times(1)
            .returning(|_, _| Box::pin(async { Ok(vec![]) }));

        let params = GetInferencesByEpisodeParams {
            function_name: Some("test_function".to_string()),
            variant_name: Some("test_variant".to_string()),
            limit: 50,
            offset: 10,
            deduplicate: false,
        };

        let result = get_inferences_by_episode(&config, &mock_clickhouse, episode_id, params)
            .await
            .unwrap();

        assert!(result.inferences.is_empty());
    }

    #[test]
    fn test_deduplicate_removes_prefix_conversations() {
        let now = Utc::now();
        let t1 = now - chrono::Duration::minutes(3);
        let t2 = now - chrono::Duration::minutes(2);
        let t3 = now - chrono::Duration::minutes(1);

        // Create three inferences with the same function and system
        // inf1: ["hello"] - prefix of inf2 and inf3
        // inf2: ["hello", "world"] - prefix of inf3
        // inf3: ["hello", "world", "!"] - leaf
        let inf1 = create_test_inference(Uuid::now_v7(), "func", Some("system"), vec!["hello"], t1);
        let inf2 = create_test_inference(
            Uuid::now_v7(),
            "func",
            Some("system"),
            vec!["hello", "world"],
            t2,
        );
        let inf3 = create_test_inference(
            Uuid::now_v7(),
            "func",
            Some("system"),
            vec!["hello", "world", "!"],
            t3,
        );

        let inferences = vec![inf1, inf2, inf3];
        let result = deduplicate_inferences(inferences);

        // Only inf3 should remain (the leaf with most messages)
        assert_eq!(result.len(), 1);
        assert_eq!(get_message_count(&result[0]), 3);
    }

    #[test]
    fn test_deduplicate_keeps_different_functions() {
        let now = Utc::now();
        let t1 = now - chrono::Duration::minutes(2);
        let t2 = now - chrono::Duration::minutes(1);

        // Two inferences with different function names - both should be kept
        let inf1 =
            create_test_inference(Uuid::now_v7(), "func_a", Some("system"), vec!["hello"], t1);
        let inf2 =
            create_test_inference(Uuid::now_v7(), "func_b", Some("system"), vec!["hello"], t2);

        let inferences = vec![inf1, inf2];
        let result = deduplicate_inferences(inferences);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_deduplicate_keeps_different_systems() {
        let now = Utc::now();
        let t1 = now - chrono::Duration::minutes(2);
        let t2 = now - chrono::Duration::minutes(1);

        // Two inferences with different system prompts - both should be kept
        let inf1 =
            create_test_inference(Uuid::now_v7(), "func", Some("system_a"), vec!["hello"], t1);
        let inf2 =
            create_test_inference(Uuid::now_v7(), "func", Some("system_b"), vec!["hello"], t2);

        let inferences = vec![inf1, inf2];
        let result = deduplicate_inferences(inferences);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_deduplicate_keeps_non_prefix_messages() {
        let now = Utc::now();
        let t1 = now - chrono::Duration::minutes(2);
        let t2 = now - chrono::Duration::minutes(1);

        // Two inferences with different messages - both should be kept
        let inf1 = create_test_inference(Uuid::now_v7(), "func", Some("system"), vec!["hello"], t1);
        let inf2 =
            create_test_inference(Uuid::now_v7(), "func", Some("system"), vec!["goodbye"], t2);

        let inferences = vec![inf1, inf2];
        let result = deduplicate_inferences(inferences);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_deduplicate_preserves_timestamp_order() {
        let now = Utc::now();
        let t1 = now - chrono::Duration::minutes(3);
        let t2 = now - chrono::Duration::minutes(2);
        let t3 = now - chrono::Duration::minutes(1);

        // Three independent conversations (different first messages)
        let inf1 = create_test_inference(Uuid::now_v7(), "func", Some("sys"), vec!["a"], t1);
        let inf2 = create_test_inference(Uuid::now_v7(), "func", Some("sys"), vec!["b"], t2);
        let inf3 = create_test_inference(Uuid::now_v7(), "func", Some("sys"), vec!["c"], t3);

        // Pass them in reverse order
        let inferences = vec![inf3.clone(), inf1.clone(), inf2.clone()];
        let result = deduplicate_inferences(inferences);

        // All should be kept and sorted by timestamp
        assert_eq!(result.len(), 3);
        assert_eq!(get_timestamp(&result[0]), t1);
        assert_eq!(get_timestamp(&result[1]), t2);
        assert_eq!(get_timestamp(&result[2]), t3);
    }

    #[test]
    fn test_deduplicate_empty_input() {
        let result = deduplicate_inferences(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_deduplicate_single_inference() {
        let now = Utc::now();
        let inf = create_test_inference(Uuid::now_v7(), "func", Some("sys"), vec!["hello"], now);

        let result = deduplicate_inferences(vec![inf]);
        assert_eq!(result.len(), 1);
    }
}
