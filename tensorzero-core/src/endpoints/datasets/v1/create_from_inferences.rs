use std::collections::HashSet;

use axum::extract::{Path, State};
use axum::Json;
use tracing::instrument;

use crate::config::Config;
use crate::db::datasets::DatasetQueries;
use crate::db::inferences::{InferenceOutputSource, InferenceQueries, ListInferencesParams};
use crate::endpoints::datasets::v1::types::CreateDatapointsFromInferenceOutputSource;
use crate::endpoints::datasets::validate_dataset_name;
use crate::error::{Error, ErrorDetails};
use crate::stored_inference::{StoredInference, StoredInferenceDatabase};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::{
    CreateDatapointsFromInferenceRequest, CreateDatapointsFromInferenceRequestParams,
    CreateDatapointsResponse,
};

/// Handler for the POST `/v1/datasets/{dataset_id}/from_inferences` endpoint.
/// Creates datapoints from inferences based on either specific inference IDs or an inference query.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "datasets.v1.create_from_inferences", skip(app_state, request))]
pub async fn create_from_inferences_handler(
    State(app_state): AppState,
    Path(dataset_name): Path<String>,
    StructuredJson(request): StructuredJson<CreateDatapointsFromInferenceRequest>,
) -> Result<Json<CreateDatapointsResponse>, Error> {
    let response = create_from_inferences(
        &app_state.config,
        &app_state.clickhouse_connection_info,
        dataset_name,
        request,
    )
    .await?;

    Ok(Json(response))
}

/// Creates datapoints from inferences based on either specific inference IDs or an inference query.
/// This happens in 2 steps:
/// 1. We query inferences table based on the request parameters (which uses list_inferences)
/// 2. We convert the inferences into datapoint_inserts, and inserts them together in up to 2 queries (one for Chat, one for Json).
pub async fn create_from_inferences(
    config: &Config,
    clickhouse: &(impl InferenceQueries + DatasetQueries),
    dataset_name: String,
    request: CreateDatapointsFromInferenceRequest,
) -> Result<CreateDatapointsResponse, Error> {
    validate_dataset_name(&dataset_name)?;

    // If output_source is not specified, default to Inference.
    let request_output_source = request
        .output_source
        .unwrap_or(CreateDatapointsFromInferenceOutputSource::Inference);

    let inference_output_source = match request_output_source {
        CreateDatapointsFromInferenceOutputSource::None => {
            // If we are not including any output in the datapoints, we use Inference for the query to
            // avoid doing a join with the DemonstrationFeedback table. Then, we will drop it when constructing the datapoints.
            InferenceOutputSource::Inference
        }
        CreateDatapointsFromInferenceOutputSource::Inference => InferenceOutputSource::Inference,
        CreateDatapointsFromInferenceOutputSource::Demonstration => {
            InferenceOutputSource::Demonstration
        }
    };

    let list_inferences_params = match &request.params {
        CreateDatapointsFromInferenceRequestParams::InferenceIds { inference_ids } => {
            ListInferencesParams {
                ids: Some(inference_ids),
                output_source: inference_output_source,
                limit: inference_ids.len() as u32,
                ..Default::default()
            }
        }
        CreateDatapointsFromInferenceRequestParams::InferenceQuery {
            function_name,
            variant_name,
            filters,
        } => ListInferencesParams {
            function_name: Some(function_name),
            variant_name: variant_name.as_deref(),
            filters: filters.as_ref(),
            output_source: inference_output_source,
            limit: u32::MAX,
            ..Default::default()
        },
    };

    // Step 1: Query inferences
    let inferences: Vec<StoredInference> = clickhouse
        .list_inferences(config, &list_inferences_params)
        .await?
        .into_iter()
        .map(StoredInferenceDatabase::into_stored_inference)
        .collect::<Result<Vec<_>, _>>()?;

    if let CreateDatapointsFromInferenceRequestParams::InferenceIds {
        inference_ids: request_inference_ids,
    } = &request.params
    {
        // Check if all inferences are found. If not, we fail early without creating any datapoints for a pseudo-transactional behavior.
        let found_inference_ids = inferences
            .iter()
            .map(StoredInference::id)
            .collect::<HashSet<_>>();
        for inference_id in request_inference_ids {
            if !found_inference_ids.contains(inference_id) {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Inference {inference_id} not found"),
                }));
            }
        }
    }

    // Convert inferences to datapoints
    let mut ids = Vec::new();
    let mut datapoints_to_insert = Vec::new();

    for inference in inferences {
        let datapoint_insert =
            inference.into_datapoint_insert(&dataset_name, &request_output_source, config)?;
        ids.push(datapoint_insert.id());
        datapoints_to_insert.push(datapoint_insert);
    }

    // Batch insert all datapoints
    if !datapoints_to_insert.is_empty() {
        clickhouse.insert_datapoints(&datapoints_to_insert).await?;
    }

    Ok(CreateDatapointsResponse { ids })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, SchemaData};
    use crate::db::clickhouse::query_builder::{InferenceFilter, TagComparisonOperator, TagFilter};
    use crate::db::clickhouse::MockClickHouseConnectionInfo;
    use crate::db::datasets::DatapointInsert;
    use crate::experimentation::ExperimentationConfig;
    use crate::function::{FunctionConfig, FunctionConfigChat, FunctionConfigJson};
    use crate::inference::types::{ContentBlockChatOutput, Text};
    use crate::jsonschema_util::StaticJSONSchema;
    use crate::stored_inference::{StoredChatInferenceDatabase, StoredInferenceDatabase};
    use crate::tool::{ToolCallConfig, ToolCallConfigDatabaseInsert, ToolChoice};
    use std::collections::HashMap;
    use std::sync::Arc;
    use uuid::Uuid;

    /// Helper to create a test config with the functions registered
    fn create_test_config() -> Config {
        let mut config = Config::default();

        // Add the test_function (Chat function)
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

        // Add the json_function (Json function)
        config.functions.insert(
            "json_function".to_string(),
            Arc::new(FunctionConfig::Json(FunctionConfigJson {
                variants: Default::default(),
                schemas: SchemaData::default(),
                output_schema: StaticJSONSchema::default(),
                json_mode_tool_call_config: ToolCallConfig::default(),
                description: None,
                experimentation: ExperimentationConfig::default(),
                all_explicit_template_names: Default::default(),
            })),
        );

        config
    }

    /// Helper to create a test inference (storage type for database)
    fn create_test_inference(id: Uuid) -> StoredInferenceDatabase {
        StoredInferenceDatabase::Chat(StoredChatInferenceDatabase {
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            input: crate::inference::types::StoredInput {
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

    #[tokio::test]
    async fn test_create_from_query_params_success() {
        let config = create_test_config();
        let function_name = "test_function";
        let variant_name = "test_variant";

        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let inference1 = create_test_inference(id1);
        let inference2 = create_test_inference(id2);

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .inference_queries
            .expect_list_inferences()
            .withf(move |_, params| {
                assert_eq!(params.function_name, Some(function_name));
                assert_eq!(params.variant_name, Some(variant_name));
                assert_eq!(params.output_source, InferenceOutputSource::Inference);
                true
            })
            .times(1)
            .returning(move |_, _| {
                let inf1 = inference1.clone();
                let inf2 = inference2.clone();
                Box::pin(async move { Ok(vec![inf1, inf2]) })
            });
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .withf(move |datapoints| {
                assert_eq!(datapoints.len(), 2, "Should insert 2 datapoints");

                true
            })
            .times(1)
            .returning(|_| Box::pin(async move { Ok(2) }));

        let request = CreateDatapointsFromInferenceRequest {
            params: CreateDatapointsFromInferenceRequestParams::InferenceQuery {
                function_name: function_name.to_string(),
                variant_name: Some(variant_name.to_string()),
                filters: None,
            },
            output_source: Some(CreateDatapointsFromInferenceOutputSource::Inference),
        };

        let result = create_from_inferences(
            &config,
            &mock_clickhouse,
            "test_dataset".to_string(),
            request,
        )
        .await
        .unwrap();

        assert_eq!(result.ids.len(), 2);
    }

    #[tokio::test]
    async fn test_create_from_query_params_returns_empty_if_no_inferences_found() {
        let config = create_test_config();
        let function_name = "test_function";
        let variant_name = "test_variant";

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .inference_queries
            .expect_list_inferences()
            .withf(move |_, params| {
                assert_eq!(params.function_name, Some(function_name));
                assert_eq!(params.variant_name, Some(variant_name));
                assert_eq!(params.output_source, InferenceOutputSource::Inference);
                true
            })
            .times(1)
            .returning(move |_, _| Box::pin(async move { Ok(vec![]) }));

        // If we didn't find any inferences, we shouldn't attempt to insert any datapoints.
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .times(0);

        let request = CreateDatapointsFromInferenceRequest {
            params: CreateDatapointsFromInferenceRequestParams::InferenceQuery {
                function_name: function_name.to_string(),
                variant_name: Some(variant_name.to_string()),
                filters: None,
            },
            output_source: Some(CreateDatapointsFromInferenceOutputSource::Inference),
        };

        let result = create_from_inferences(
            &config,
            &mock_clickhouse,
            "test_dataset".to_string(),
            request,
        )
        .await
        .unwrap();

        assert!(result.ids.is_empty(), "Expected no datapoints returned");
    }

    #[tokio::test]
    async fn test_create_from_query_params_queries_for_demonstration_output() {
        let config = create_test_config();
        let function_name = "test_function";
        let variant_name = "test_variant";

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .inference_queries
            .expect_list_inferences()
            .withf(move |_, params| {
                assert_eq!(params.output_source, InferenceOutputSource::Demonstration);
                true
            })
            .times(1)
            .returning(move |_, _| Box::pin(async move { Ok(vec![]) }));
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .times(0);

        let request = CreateDatapointsFromInferenceRequest {
            params: CreateDatapointsFromInferenceRequestParams::InferenceQuery {
                function_name: function_name.to_string(),
                variant_name: Some(variant_name.to_string()),
                filters: None,
            },
            output_source: Some(CreateDatapointsFromInferenceOutputSource::Demonstration),
        };

        create_from_inferences(
            &config,
            &mock_clickhouse,
            "test_dataset".to_string(),
            request,
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_create_from_inference_ids_success() {
        let config = create_test_config();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();

        let inference1 = create_test_inference(id1);
        let inference2 = create_test_inference(id2);

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .inference_queries
            .expect_list_inferences()
            .withf(move |_, params| {
                let requested_ids = params.ids.unwrap();
                requested_ids.contains(&id1) && requested_ids.contains(&id2)
            })
            .times(1)
            .returning(move |_, _| {
                let inf1 = inference1.clone();
                let inf2 = inference2.clone();
                Box::pin(async move { Ok(vec![inf1, inf2]) })
            });
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .withf(move |datapoints| {
                assert_eq!(datapoints.len(), 2, "Should insert 2 datapoints");

                true
            })
            .times(1)
            .returning(|_| Box::pin(async move { Ok(2) }));

        let request = CreateDatapointsFromInferenceRequest {
            params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
                inference_ids: vec![id1, id2],
            },
            output_source: Some(CreateDatapointsFromInferenceOutputSource::Inference),
        };

        let result = create_from_inferences(
            &config,
            &mock_clickhouse,
            "test_dataset".to_string(),
            request,
        )
        .await
        .unwrap();

        assert_eq!(result.ids.len(), 2);
    }

    #[tokio::test]
    async fn test_create_from_inference_ids_missing_inference_fails() {
        let config = create_test_config();
        let existing_id = Uuid::now_v7();
        let missing_id = Uuid::now_v7();

        let inference = create_test_inference(existing_id);

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .inference_queries
            .expect_list_inferences()
            .times(1)
            .returning(move |_, _| {
                let inf = inference.clone();
                Box::pin(async move { Ok(vec![inf]) })
            });
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .times(0);

        let request = CreateDatapointsFromInferenceRequest {
            params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
                inference_ids: vec![existing_id, missing_id],
            },
            output_source: Some(CreateDatapointsFromInferenceOutputSource::Inference),
        };

        let result = create_from_inferences(
            &config,
            &mock_clickhouse,
            "test_dataset".to_string(),
            request,
        )
        .await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(
            error.to_string().contains("not found"),
            "Expected 'not found' error, got: {error}"
        );
    }

    #[tokio::test]
    async fn test_invalid_dataset_name_fails_without_querying() {
        let config = create_test_config();

        // We shouldn't query inferences or insert datapoints if the dataset name is invalid.
        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .inference_queries
            .expect_list_inferences()
            .times(0);
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .times(0);

        let request = CreateDatapointsFromInferenceRequest {
            params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
                inference_ids: vec![Uuid::now_v7()],
            },
            output_source: Some(CreateDatapointsFromInferenceOutputSource::Inference),
        };

        // Dataset name "builder" is reserved
        let result =
            create_from_inferences(&config, &mock_clickhouse, "builder".to_string(), request).await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(
            error.to_string().contains("Invalid") || error.to_string().contains("dataset"),
            "Expected dataset validation error, got: {error}"
        );
    }

    #[tokio::test]
    async fn test_output_source_none_drops_output() {
        let config = create_test_config();
        let id = Uuid::now_v7();

        let inference = create_test_inference(id);

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .inference_queries
            .expect_list_inferences()
            .times(1)
            .returning(move |_, _| {
                let inf = inference.clone();
                Box::pin(async move { Ok(vec![inf]) })
            });
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .times(1)
            .withf(|datapoints| {
                // Verify that the datapoint has no output when output_source is None
                let Some(DatapointInsert::Chat(dp)) = datapoints.first() else {
                    panic!("Expected a chat datapoint")
                };
                assert!(dp.output.is_none(), "Datapoint output should be dropped");
                true
            })
            .returning(|_| Box::pin(async move { Ok(1) }));

        let request = CreateDatapointsFromInferenceRequest {
            params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
                inference_ids: vec![id],
            },
            output_source: Some(CreateDatapointsFromInferenceOutputSource::None),
        };

        let result = create_from_inferences(
            &config,
            &mock_clickhouse,
            "test_dataset".to_string(),
            request,
        )
        .await
        .unwrap();

        assert_eq!(result.ids.len(), 1);
    }

    #[tokio::test]
    async fn test_output_source_inference_preserves_output() {
        let config = create_test_config();
        let id = Uuid::now_v7();

        let inference = create_test_inference(id);

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .inference_queries
            .expect_list_inferences()
            .times(1)
            .returning(move |_, _| {
                let inf = inference.clone();
                Box::pin(async move { Ok(vec![inf]) })
            });
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .times(1)
            .withf(|datapoints| {
                // Verify that the datapoint has the output when output_source is Inference
                let Some(DatapointInsert::Chat(dp)) = datapoints.first() else {
                    panic!("Expected a chat datapoint")
                };
                assert_eq!(
                    dp.output.as_ref().unwrap(),
                    &vec![ContentBlockChatOutput::Text(Text {
                        text: "test output".to_string(),
                    })],
                    "Datapoint output should be preserved"
                );
                true
            })
            .returning(|_| Box::pin(async move { Ok(1) }));

        let request = CreateDatapointsFromInferenceRequest {
            params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
                inference_ids: vec![id],
            },
            output_source: Some(CreateDatapointsFromInferenceOutputSource::Inference),
        };

        let result = create_from_inferences(
            &config,
            &mock_clickhouse,
            "test_dataset".to_string(),
            request,
        )
        .await
        .unwrap();

        assert_eq!(result.ids.len(), 1);
    }

    #[tokio::test]
    async fn test_filters_passed_to_list_inferences() {
        let config = create_test_config();
        let function_name = "test_function";
        let tag_key = "environment";
        let tag_value = "production";

        let id = Uuid::now_v7();
        let inference = create_test_inference(id);

        // Create a filter to test
        let test_filter = InferenceFilter::Tag(TagFilter {
            key: tag_key.to_string(),
            value: tag_value.to_string(),
            comparison_operator: TagComparisonOperator::Equal,
        });

        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
        mock_clickhouse
            .inference_queries
            .expect_list_inferences()
            .withf(move |_, params| {
                // Verify function_name is passed correctly
                assert_eq!(params.function_name, Some(function_name));

                // Verify filters are passed correctly
                assert!(params.filters.is_some(), "Filters should be passed through");

                if let Some(filter) = params.filters {
                    // Verify the filter structure
                    if let InferenceFilter::Tag(tag_filter) = filter {
                        assert_eq!(tag_filter.key, tag_key);
                        assert_eq!(tag_filter.value, tag_value);
                        assert_eq!(tag_filter.comparison_operator, TagComparisonOperator::Equal);
                    } else {
                        panic!("Expected TagFilter");
                    }
                }

                true
            })
            .times(1)
            .returning(move |_, _| {
                let inf = inference.clone();
                Box::pin(async move { Ok(vec![inf]) })
            });

        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
            .withf(|datapoints| datapoints.len() == 1)
            .times(1)
            .returning(|_| Box::pin(async move { Ok(1) }));

        let request = CreateDatapointsFromInferenceRequest {
            params: CreateDatapointsFromInferenceRequestParams::InferenceQuery {
                function_name: function_name.to_string(),
                variant_name: None,
                filters: Some(test_filter),
            },
            output_source: Some(CreateDatapointsFromInferenceOutputSource::Inference),
        };

        let result = create_from_inferences(
            &config,
            &mock_clickhouse,
            "test_dataset".to_string(),
            request,
        )
        .await
        .unwrap();

        assert_eq!(result.ids.len(), 1);
    }
}
