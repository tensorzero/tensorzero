use axum::extract::State;
use axum::Json;
use tracing::instrument;

use crate::config::Config;
use crate::db::inferences::{InferenceQueries, ListInferencesParams};
use crate::error::Error;
use crate::stored_inference::StoredInferenceDatabase;
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::{GetInferencesRequest, GetInferencesResponse, ListInferencesRequest};

/// Handler for the POST `/v1/inferences/get_inferences` endpoint.
/// Retrieves specific inferences by their IDs.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "inferences.v1.get_inferences", skip(app_state, request))]
pub async fn get_inferences_handler(
    State(app_state): AppState,
    StructuredJson(request): StructuredJson<GetInferencesRequest>,
) -> Result<Json<GetInferencesResponse>, Error> {
    let response = get_inferences(
        &app_state.config,
        &app_state.clickhouse_connection_info,
        request,
    )
    .await?;
    Ok(Json(response))
}

pub async fn get_inferences(
    config: &Config,
    clickhouse: &impl InferenceQueries,
    request: GetInferencesRequest,
) -> Result<GetInferencesResponse, Error> {
    // If no IDs are provided, return an empty response.
    if request.ids.is_empty() {
        return Ok(GetInferencesResponse { inferences: vec![] });
    }

    // TODO(shuyangli): Consider restricting the number of inferences to return to avoid unbounded queries.
    let params = ListInferencesParams {
        ids: Some(&request.ids),
        function_name: request.function_name.as_deref(),
        output_source: request.output_source,
        limit: u32::MAX,
        offset: 0,
        ..Default::default()
    };

    let inferences_storage = clickhouse.list_inferences(config, &params).await?;
    let inferences = inferences_storage
        .into_iter()
        .map(StoredInferenceDatabase::into_stored_inference)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(GetInferencesResponse { inferences })
}

/// Handler for the POST `/v1/inferences/list_inferences` endpoint.
/// Lists inferences with optional filtering, pagination, and sorting.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "inferences.v1.list_inferences", skip(app_state, request))]
pub async fn list_inferences_handler(
    State(app_state): AppState,
    StructuredJson(request): StructuredJson<ListInferencesRequest>,
) -> Result<Json<GetInferencesResponse>, Error> {
    let response = list_inferences(
        &app_state.config,
        &app_state.clickhouse_connection_info,
        request,
    )
    .await?;

    Ok(Json(response))
}

pub async fn list_inferences(
    config: &Config,
    clickhouse: &impl InferenceQueries,
    request: ListInferencesRequest,
) -> Result<GetInferencesResponse, Error> {
    let params = request.as_list_inferences_params();
    let inferences_storage = clickhouse.list_inferences(config, &params).await?;
    let inferences = inferences_storage
        .into_iter()
        .map(StoredInferenceDatabase::into_stored_inference)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(GetInferencesResponse { inferences })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, SchemaData};
    use crate::db::inferences::{
        InferenceOutputSource, MockInferenceQueries, DEFAULT_INFERENCE_QUERY_LIMIT,
    };
    use crate::experimentation::ExperimentationConfig;
    use crate::function::{FunctionConfig, FunctionConfigChat};
    use crate::inference::types::{ContentBlockChatOutput, StoredInput, Text};
    use crate::stored_inference::{
        StoredChatInferenceDatabase, StoredInference, StoredInferenceDatabase,
    };
    use crate::tool::{ToolCallConfigDatabaseInsert, ToolChoice};
    use std::collections::HashMap;
    use std::sync::Arc;
    use uuid::Uuid;

    /// Helper to create a test config with the functions registered
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

    /// Helper to create a test inference (storage type for database)
    fn create_test_inference_database(id: Uuid) -> StoredInferenceDatabase {
        StoredInferenceDatabase::Chat(StoredChatInferenceDatabase {
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            input: StoredInput {
                system: None,
                messages: vec![],
            },
            output: vec![ContentBlockChatOutput::Text(Text {
                text: "test output".to_string(),
            })],
            dispreferred_outputs: vec![],
            timestamp: chrono::Utc::now(),
            episode_id: Uuid::now_v7(),
            inference_id: id,
            tool_params: ToolCallConfigDatabaseInsert::default(),
            tags: HashMap::new(),
        })
    }

    // Tests for get_inferences()

    #[tokio::test]
    async fn test_get_inferences_with_valid_ids() {
        let config = create_test_config();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let ids = vec![id1, id2];

        let inference1 = create_test_inference_database(id1);
        let inference2 = create_test_inference_database(id2);

        let mut mock_clickhouse = MockInferenceQueries::new();
        mock_clickhouse
            .expect_list_inferences()
            .withf(move |_, params| {
                // Verify correct parameters
                assert_eq!(params.ids, Some(ids.as_slice()));
                assert_eq!(params.output_source, InferenceOutputSource::Inference);
                assert_eq!(params.limit, u32::MAX);
                assert_eq!(params.offset, 0);
                assert_eq!(params.function_name, None);
                assert_eq!(params.variant_name, None);
                assert_eq!(params.episode_id, None);
                assert!(params.filters.is_none());
                assert!(params.order_by.is_none());
                true
            })
            .times(1)
            .returning(move |_, _| {
                let inf1 = inference1.clone();
                let inf2 = inference2.clone();
                Box::pin(async move { Ok(vec![inf1, inf2]) })
            });

        let request = GetInferencesRequest {
            ids: vec![id1, id2],
            function_name: None,
            output_source: InferenceOutputSource::Inference,
        };

        let result = get_inferences(&config, &mock_clickhouse, request)
            .await
            .unwrap();

        assert_eq!(result.inferences.len(), 2);
    }

    #[tokio::test]
    async fn test_get_inferences_with_empty_ids() {
        let config = create_test_config();
        let mut mock_clickhouse = MockInferenceQueries::new();

        // Should NOT call list_inferences when IDs are empty
        mock_clickhouse.expect_list_inferences().times(0);

        let request = GetInferencesRequest {
            ids: vec![],
            function_name: None,
            output_source: InferenceOutputSource::Inference,
        };

        let result = get_inferences(&config, &mock_clickhouse, request)
            .await
            .unwrap();

        assert_eq!(result.inferences.len(), 0);
    }

    #[tokio::test]
    async fn test_get_inferences_converts_to_wire_type() {
        let config = create_test_config();
        let id = Uuid::now_v7();
        let inference = create_test_inference_database(id);

        let mut mock_clickhouse = MockInferenceQueries::new();
        mock_clickhouse
            .expect_list_inferences()
            .times(1)
            .returning(move |_, _| {
                let inf = inference.clone();
                Box::pin(async move { Ok(vec![inf]) })
            });

        let request = GetInferencesRequest {
            ids: vec![id],
            function_name: None,
            output_source: InferenceOutputSource::Inference,
        };

        let result = get_inferences(&config, &mock_clickhouse, request)
            .await
            .unwrap();

        // Verify it returns StoredInference (wire type), not StoredInferenceDatabase
        assert_eq!(result.inferences.len(), 1);
        // Wire type should have the same basic properties
        let StoredInference::Chat(ref inference) = result.inferences[0] else {
            panic!("Expected Chat inference");
        };
        assert_eq!(inference.function_name, "test_function");
        assert_eq!(inference.variant_name, "test_variant");
        assert_eq!(inference.inference_id, id);
    }

    #[tokio::test]
    async fn test_get_inferences_uses_correct_params() {
        let config = create_test_config();
        let id = Uuid::now_v7();
        let inference = create_test_inference_database(id);

        let mut mock_clickhouse = MockInferenceQueries::new();
        mock_clickhouse
            .expect_list_inferences()
            .withf(move |_, params| {
                // Specifically verify limit and output_source
                assert_eq!(params.limit, u32::MAX, "Should use u32::MAX limit");
                assert_eq!(
                    params.output_source,
                    InferenceOutputSource::Demonstration,
                    "Should use Inference output source"
                );
                true
            })
            .times(1)
            .returning(move |_, _| {
                let inf = inference.clone();
                Box::pin(async move { Ok(vec![inf]) })
            });

        let request = GetInferencesRequest {
            ids: vec![id],
            function_name: None,
            output_source: InferenceOutputSource::Demonstration,
        };

        let result = get_inferences(&config, &mock_clickhouse, request).await;
        assert!(result.is_ok());
    }

    // Tests for list_inferences()

    #[tokio::test]
    async fn test_list_inferences_with_defaults() {
        let config = create_test_config();
        let id = Uuid::now_v7();
        let inference = create_test_inference_database(id);

        let mut mock_clickhouse = MockInferenceQueries::new();
        mock_clickhouse
            .expect_list_inferences()
            .withf(|_, params| {
                // Verify default pagination values
                assert_eq!(
                    params.limit, DEFAULT_INFERENCE_QUERY_LIMIT,
                    "Should enforce a default limit"
                );
                assert_eq!(params.ids, None);
                true
            })
            .times(1)
            .returning(move |_, _| {
                let inf = inference.clone();
                Box::pin(async move { Ok(vec![inf]) })
            });

        let request = ListInferencesRequest {
            output_source: InferenceOutputSource::Inference,
            ..Default::default()
        };

        let result = list_inferences(&config, &mock_clickhouse, request)
            .await
            .unwrap();

        assert_eq!(result.inferences.len(), 1);
    }

    #[tokio::test]
    async fn test_list_inferences_with_custom_pagination() {
        let config = create_test_config();
        let id = Uuid::now_v7();
        let inference = create_test_inference_database(id);

        let mut mock_clickhouse = MockInferenceQueries::new();
        mock_clickhouse
            .expect_list_inferences()
            .withf(|_, params| {
                // Verify custom pagination values
                assert_eq!(params.limit, 50, "Should use custom limit");
                assert_eq!(params.offset, 100, "Should use custom offset");
                true
            })
            .times(1)
            .returning(move |_, _| {
                let inf = inference.clone();
                Box::pin(async move { Ok(vec![inf]) })
            });

        let request = ListInferencesRequest {
            output_source: InferenceOutputSource::Inference,
            limit: Some(50),
            offset: Some(100),
            ..Default::default()
        };

        let result = list_inferences(&config, &mock_clickhouse, request)
            .await
            .unwrap();

        assert_eq!(result.inferences.len(), 1);
    }

    #[tokio::test]
    async fn test_list_inferences_with_filters() {
        let config = create_test_config();
        let id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let inference = create_test_inference_database(id);

        let function_name = "test_function".to_string();
        let variant_name = "test_variant".to_string();

        let mut mock_clickhouse = MockInferenceQueries::new();
        mock_clickhouse
            .expect_list_inferences()
            .withf(move |_, params| {
                // Verify filters are passed correctly
                assert_eq!(
                    params.function_name,
                    Some("test_function"),
                    "Should pass function_name"
                );
                assert_eq!(
                    params.variant_name,
                    Some("test_variant"),
                    "Should pass variant_name"
                );
                assert_eq!(
                    params.episode_id,
                    Some(&episode_id),
                    "Should pass episode_id"
                );
                true
            })
            .times(1)
            .returning(move |_, _| {
                let inf = inference.clone();
                Box::pin(async move { Ok(vec![inf]) })
            });

        let request = ListInferencesRequest {
            function_name: Some(function_name),
            variant_name: Some(variant_name),
            episode_id: Some(episode_id),
            output_source: InferenceOutputSource::Inference,
            ..Default::default()
        };

        let result = list_inferences(&config, &mock_clickhouse, request)
            .await
            .unwrap();

        assert_eq!(result.inferences.len(), 1);
    }

    #[tokio::test]
    async fn test_list_inferences_with_order_by() {
        let config = create_test_config();
        let id = Uuid::now_v7();
        let inference = create_test_inference_database(id);

        use crate::db::clickhouse::query_builder::{OrderBy, OrderByTerm, OrderDirection};

        let order_by = vec![OrderBy {
            term: OrderByTerm::Timestamp,
            direction: OrderDirection::Desc,
        }];

        let mut mock_clickhouse = MockInferenceQueries::new();
        mock_clickhouse
            .expect_list_inferences()
            .withf(move |_, params| {
                // Verify order_by is passed correctly
                assert!(params.order_by.is_some(), "Should have order_by");
                assert_eq!(params.order_by.unwrap().len(), 1);
                true
            })
            .times(1)
            .returning(move |_, _| {
                let inf = inference.clone();
                Box::pin(async move { Ok(vec![inf]) })
            });

        let request = ListInferencesRequest {
            output_source: InferenceOutputSource::Inference,
            order_by: Some(order_by),
            ..Default::default()
        };

        let result = list_inferences(&config, &mock_clickhouse, request)
            .await
            .unwrap();

        assert_eq!(result.inferences.len(), 1);
    }

    #[tokio::test]
    async fn test_list_inferences_empty_results() {
        let config = create_test_config();

        let mut mock_clickhouse = MockInferenceQueries::new();
        mock_clickhouse
            .expect_list_inferences()
            .times(1)
            .returning(|_, _| Box::pin(async move { Ok(vec![]) }));

        let request = ListInferencesRequest {
            output_source: InferenceOutputSource::Inference,
            ..Default::default()
        };

        let result = list_inferences(&config, &mock_clickhouse, request)
            .await
            .unwrap();

        assert_eq!(result.inferences.len(), 0);
    }
}
