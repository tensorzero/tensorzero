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
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::{
    CreateDatapointsFromInferenceRequest, CreateDatapointsFromInferenceRequestParams,
    CreateDatapointsFromInferenceResponse,
};

/// Handler for the POST `/v1/datasets/{dataset_id}/from_inferences` endpoint.
/// Creates datapoints from inferences based on either specific inference IDs or an inference query.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "datasets.v1.create_from_inferences", skip(app_state, request))]
pub async fn create_from_inferences_handler(
    State(app_state): AppState,
    Path(dataset_name): Path<String>,
    StructuredJson(request): StructuredJson<CreateDatapointsFromInferenceRequest>,
) -> Result<Json<CreateDatapointsFromInferenceResponse>, Error> {
    let response = create_from_inferences(
        &app_state.config,
        &app_state.clickhouse_connection_info,
        dataset_name,
        request,
    )
    .await?;

    Ok(Json(response))
}

async fn create_from_inferences(
    config: &Config,
    clickhouse: &(impl InferenceQueries + DatasetQueries),
    dataset_name: String,
    request: CreateDatapointsFromInferenceRequest,
) -> Result<CreateDatapointsFromInferenceResponse, Error> {
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
            ..Default::default()
        },
    };
    let inferences = clickhouse
        .list_inferences(config, &list_inferences_params)
        .await?;

    if let CreateDatapointsFromInferenceRequestParams::InferenceIds {
        inference_ids: request_inference_ids,
    } = &request.params
    {
        // Check if all inferences are found. If not, we fail early without creating any datapoints for a pseudo-transactional behavior.
        let found_inference_ids = inferences
            .iter()
            .map(crate::stored_inference::StoredInference::id)
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
            inference.into_datapoint_insert(&dataset_name, &request_output_source);
        ids.push(datapoint_insert.id());
        datapoints_to_insert.push(datapoint_insert);
    }

    // Batch insert all datapoints
    if !datapoints_to_insert.is_empty() {
        clickhouse.insert_datapoints(&datapoints_to_insert).await?;
    }

    Ok(CreateDatapointsFromInferenceResponse { ids })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::gateway::GatewayConfig;
    use crate::config::provider_types::ProviderTypesConfig;
    use crate::config::PostgresConfig;
    use crate::db::clickhouse::MockClickHouseConnectionInfo;
    use crate::db::datasets::DatapointInsert;
    use crate::embeddings::EmbeddingModelTable;
    use crate::inference::types::{ContentBlockChatOutput, Text};
    use crate::minijinja_util::TemplateConfig;
    use crate::model::ModelTable;
    use crate::rate_limiting::RateLimitingConfig;
    use crate::stored_inference::{StoredChatInference, StoredInference};
    use crate::tool::ToolCallConfigDatabaseInsert;
    use std::collections::HashMap;
    use std::sync::Arc;
    use uuid::Uuid;

    /// Helper to create a minimal test config
    fn create_test_config() -> Config {
        Config {
            gateway: GatewayConfig::default(),
            models: Arc::new(ModelTable::default()),
            embedding_models: Arc::new(EmbeddingModelTable::default()),
            functions: HashMap::new(),
            metrics: HashMap::new(),
            tools: HashMap::new(),
            evaluations: HashMap::new(),
            templates: Arc::new(TemplateConfig::new()),
            object_store_info: None,
            provider_types: ProviderTypesConfig::default(),
            optimizers: HashMap::new(),
            postgres: PostgresConfig::default(),
            rate_limiting: RateLimitingConfig::default(),
        }
    }

    /// Helper to create a test inference
    fn create_test_inference(id: Uuid) -> StoredInference {
        StoredInference::Chat(StoredChatInference {
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
            .times(1)
            .returning(move |_, _| {
                let inf1 = inference1.clone();
                let inf2 = inference2.clone();
                Box::pin(async move { Ok(vec![inf1, inf2]) })
            });
        mock_clickhouse
            .dataset_queries
            .expect_insert_datapoints()
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
    async fn test_invalid_dataset_name() {
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
    async fn test_output_source_none() {
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
                if let Some(DatapointInsert::Chat(dp)) = datapoints.first() {
                    dp.output.is_none()
                } else {
                    false
                }
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
}
