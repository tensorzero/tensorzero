use serde_json::json;
use std::collections::{HashMap, HashSet};
use tensorzero_core::db::clickhouse::migration_manager::{self, RunMigrationManagerArgs};
use tensorzero_core::endpoints::datasets::v1::types::{DatapointOrderBy, DatapointOrderByTerm};
use uuid::Uuid;

use object_store::path::Path as ObjectStorePath;
use tensorzero::{
    DatasetQueryParams, FloatComparisonOperator, GetDatapointParams, GetDatasetMetadataParams,
    OrderDirection, Role, StoredDatapoint,
};
use tensorzero_core::config::{MetricConfigLevel, MetricConfigType};
use tensorzero_core::db::clickhouse::test_helpers::{
    clickhouse_flush_async_insert, get_clickhouse,
};
use tensorzero_core::db::datasets::{
    ChatInferenceDatapointInsert, CountDatapointsForDatasetFunctionParams, DatapointInsert,
    DatasetMetadata, DatasetOutputSource, DatasetQueries, GetDatapointsParams,
    JsonInferenceDatapointInsert, MetricFilter,
};
use tensorzero_core::endpoints::datasets::DatapointKind;
use tensorzero_core::inference::types::file::ObjectStoragePointer;
use tensorzero_core::inference::types::storage::{StorageKind, StoragePath};
use tensorzero_core::inference::types::stored_input::StoredFile;
use tensorzero_core::inference::types::{
    ContentBlockChatOutput, JsonInferenceOutput, StoredInput, StoredInputMessage,
    StoredInputMessageContent, Text,
};
use tensorzero_core::stored_inference::StoredSample;

use crate::clickhouse::get_clean_clickhouse;

#[tokio::test]
async fn test_count_rows_for_chat_dataset_with_write_haiku_function() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Chat,
        function_name: Some("write_haiku".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: None,
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(count > 0, "Should have existing chat inferences");
}

#[tokio::test]
async fn test_count_rows_for_json_dataset_with_extract_entities_function_and_variant() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Json,
        function_name: Some("extract_entities".to_string()),
        dataset_name: None,
        variant_name: Some("llama_8b_initial_prompt".to_string()),
        extra_where: None,
        extra_params: None,
        metric_filter: None,
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(count > 0, "Should have existing json inferences");
}

#[tokio::test]
async fn test_count_rows_with_float_metric_filter() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Chat,
        function_name: Some("write_haiku".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "haiku_rating".to_string(),
            metric_type: MetricConfigType::Float,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.8,
            join_on: MetricConfigLevel::Inference,
        }),
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(
        count > 0,
        "Should have existing inferences with float metric filter"
    );
}

#[tokio::test]
async fn test_count_rows_chat_datapoints_with_boolean_metric_filter() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Chat,
        function_name: Some("write_haiku".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "haiku_score".to_string(),
            metric_type: MetricConfigType::Boolean,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.0,
            join_on: MetricConfigLevel::Inference,
        }),
        output_source: DatasetOutputSource::Inference,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(
        count > 0,
        "Should have existing inferences with boolean metric filter"
    );
}

#[tokio::test]
async fn test_count_rows_json_datapoints_with_boolean_metric_filter() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Json,
        function_name: Some("extract_entities".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "exact_match".to_string(),
            metric_type: MetricConfigType::Boolean,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.0,
            join_on: MetricConfigLevel::Inference,
        }),
        output_source: DatasetOutputSource::Inference,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(
        count > 0,
        "Should have existing inferences with boolean metric filter"
    );
}

#[tokio::test]
async fn test_count_rows_chat_datapoints_with_boolean_metric_filter_at_episode_level() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Chat,
        function_name: Some("write_haiku".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "haiku_score_episode".to_string(),
            metric_type: MetricConfigType::Boolean,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.0,
            join_on: MetricConfigLevel::Episode,
        }),
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(
        count > 0,
        "Should have existing inferences with boolean metric filter"
    );
}

#[tokio::test]
async fn test_count_rows_json_datapoints_with_boolean_metric_filter_at_episode_level() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Json,
        function_name: Some("extract_entities".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "exact_match_episode".to_string(),
            metric_type: MetricConfigType::Boolean,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.0,
            join_on: MetricConfigLevel::Episode,
        }),
        output_source: DatasetOutputSource::Inference,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(
        count > 0,
        "Should have existing inferences with boolean metric filter"
    );
}

#[tokio::test]
async fn test_count_rows_chat_datapoints_with_float_metric_filter_at_inference_level() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Chat,
        function_name: Some("write_haiku".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "haiku_rating".to_string(),
            metric_type: MetricConfigType::Float,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.8,
            join_on: MetricConfigLevel::Inference,
        }),
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(
        count > 0,
        "Should have existing inferences with float metric filter"
    );
}

#[tokio::test]
async fn test_count_rows_json_datapoints_with_float_metric_filter_at_inference_level() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Json,
        function_name: Some("extract_entities".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "jaccard_similarity".to_string(),
            metric_type: MetricConfigType::Float,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.8,
            join_on: MetricConfigLevel::Inference,
        }),
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(
        count > 0,
        "Should have existing inferences with float metric filter"
    );
}

#[tokio::test]
async fn test_count_rows_chat_datapoints_with_float_metric_filter_at_episode_level() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Chat,
        function_name: Some("write_haiku".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "haiku_rating_episode".to_string(),
            metric_type: MetricConfigType::Float,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.8,
            join_on: MetricConfigLevel::Episode,
        }),
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(
        count > 0,
        "Should have existing inferences with float metric filter"
    );
}

#[tokio::test]
async fn test_count_rows_json_datapoints_with_float_metric_filter_at_episode_level() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Json,
        function_name: Some("extract_entities".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "jaccard_similarity_episode".to_string(),
            metric_type: MetricConfigType::Float,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.8,
            join_on: MetricConfigLevel::Episode,
        }),
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(
        count > 0,
        "Should have existing inferences with float metric filter"
    );
}

#[tokio::test]
async fn test_count_rows_chat_datapoints_with_metric_filter_and_demonstration_join() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Chat,
        function_name: Some("write_haiku".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "haiku_rating".to_string(),
            metric_type: MetricConfigType::Float,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.8,
            join_on: MetricConfigLevel::Inference,
        }),
        output_source: DatasetOutputSource::Demonstration,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(
        count > 0,
        "Should have existing inferences with float metric filter"
    );
}

#[tokio::test]
async fn test_count_rows_json_datapoints_with_float_metric_filter_and_demonstration_join() {
    let params = DatasetQueryParams {
        inference_type: DatapointKind::Json,
        function_name: Some("extract_entities".to_string()),
        dataset_name: None,
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: Some(MetricFilter {
            metric: "jaccard_similarity".to_string(),
            metric_type: MetricConfigType::Float,
            operator: FloatComparisonOperator::GreaterThan,
            threshold: 0.8,
            join_on: MetricConfigLevel::Inference,
        }),
        output_source: DatasetOutputSource::Demonstration,
        limit: None,
        offset: None,
    };

    let count = get_clickhouse()
        .await
        .count_rows_for_dataset(&params)
        .await
        .unwrap();

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert_eq!(count, 0, "Should have 0 inferences");
}

#[tokio::test]
async fn test_get_dataset_metadata_returns_correct_counts_for_all_datasets() {
    let params = GetDatasetMetadataParams {
        function_name: None,
        limit: None,
        offset: None,
    };
    let metadata = get_clickhouse()
        .await
        .get_dataset_metadata(&params)
        .await
        .unwrap();

    // We only assert that the result contains the expected datasets
    // Because other tests insert into the table, there could be additional datasets
    assert!(metadata.contains(&DatasetMetadata {
        dataset_name: "foo".to_string(),
        count: 118,
        last_updated: "2025-04-15T02:33:58Z".to_string(),
    }));
    assert!(metadata.contains(&DatasetMetadata {
        dataset_name: "bar".to_string(),
        count: 6,
        last_updated: "2025-03-14T17:38:09Z".to_string(),
    }));
}

#[tokio::test]
async fn test_get_dataset_metadata_returns_correct_counts_for_specific_function() {
    let params = GetDatasetMetadataParams {
        function_name: Some("write_haiku".to_string()),
        limit: None,
        offset: None,
    };
    let metadata = get_clickhouse()
        .await
        .get_dataset_metadata(&params)
        .await
        .unwrap();

    // We only assert that the result contains the expected dataset
    // Because other tests insert into the table, there could be additional datasets
    assert!(metadata.contains(&DatasetMetadata {
        dataset_name: "foo".to_string(),
        count: 77,
        last_updated: "2025-03-23T20:03:59Z".to_string(),
    }));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_clickhouse_count_datasets() {
    let (clickhouse, _guard) = get_clean_clickhouse(false).await;
    let is_manual = clickhouse.is_cluster_configured();
    migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: is_manual,
        disable_automatic_migrations: false,
    })
    .await
    .unwrap();

    // Get initial count
    let initial_count = clickhouse.count_datasets().await.unwrap();
    assert_eq!(initial_count, 0, "Should have 0 datasets before insertion");

    // Insert datapoints in two different datasets
    let dataset1 = format!("test_dataset_{}", Uuid::now_v7());
    let dataset2 = format!("test_dataset_{}", Uuid::now_v7());

    let datapoint1 = ChatInferenceDatapointInsert {
        dataset_name: dataset1.clone(),
        function_name: "test_function".to_string(),
        id: Uuid::now_v7(),
        name: None,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "test".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    let datapoint2 = ChatInferenceDatapointInsert {
        dataset_name: dataset2.clone(),
        function_name: "test_function".to_string(),
        id: Uuid::now_v7(),
        name: None,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "test".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    clickhouse
        .insert_datapoints(&[DatapointInsert::Chat(datapoint1)])
        .await
        .unwrap();
    clickhouse
        .insert_datapoints(&[DatapointInsert::Chat(datapoint2)])
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Count should equal 2
    let new_count = clickhouse.count_datasets().await.unwrap();
    assert_eq!(new_count, 2);
}

#[tokio::test]
async fn test_count_datapoints_for_dataset_function_chat() {
    let clickhouse = get_clickhouse().await;

    let dataset_name = format!("test_count_{}", Uuid::now_v7());
    let function_name = "test_function";

    // Get initial count
    let initial_count = clickhouse
        .count_datapoints_for_dataset_function(&CountDatapointsForDatasetFunctionParams {
            dataset_name: dataset_name.clone(),
            function_name: function_name.to_string(),
            function_type: DatapointKind::Chat,
        })
        .await
        .unwrap();
    assert_eq!(
        initial_count, 0,
        "Newly-created dataset should have 0 datapoints before insertion"
    );

    // Insert two datapoints
    for _ in 0..2 {
        let datapoint = ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: function_name.to_string(),
            id: Uuid::now_v7(),
            name: None,
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "test".to_string(),
            })]),
            tool_params: None,
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };

        clickhouse
            .insert_datapoints(&[DatapointInsert::Chat(datapoint)])
            .await
            .unwrap();
    }

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let new_count = clickhouse
        .count_datapoints_for_dataset_function(&CountDatapointsForDatasetFunctionParams {
            dataset_name: dataset_name.clone(),
            function_name: function_name.to_string(),
            function_type: DatapointKind::Chat,
        })
        .await
        .unwrap();

    assert_eq!(
        new_count, 2,
        "Dataset should have 2 chat datapoints after insertion"
    );
}

#[tokio::test]
async fn test_count_datapoints_for_dataset_function_json() {
    let clickhouse = get_clickhouse().await;

    let dataset_name = format!("test_count_{}", Uuid::now_v7());
    let function_name = "test_function";

    // Get initial count
    let initial_count = clickhouse
        .count_datapoints_for_dataset_function(&CountDatapointsForDatasetFunctionParams {
            dataset_name: dataset_name.clone(),
            function_name: function_name.to_string(),
            function_type: DatapointKind::Json,
        })
        .await
        .unwrap();
    assert_eq!(
        initial_count, 0,
        "Newly-created dataset should have 0 datapoints before insertion"
    );

    // Insert two datapoints
    for _ in 0..2 {
        let datapoint = JsonInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: function_name.to_string(),
            id: Uuid::now_v7(),
            name: None,
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![],
            },
            output: Some(JsonInferenceOutput {
                raw: None,
                parsed: None,
            }),
            output_schema: json!({"type":"object"}),
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };

        clickhouse
            .insert_datapoints(&[DatapointInsert::Json(datapoint)])
            .await
            .unwrap();
    }

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let new_count = clickhouse
        .count_datapoints_for_dataset_function(&CountDatapointsForDatasetFunctionParams {
            dataset_name: dataset_name.clone(),
            function_name: function_name.to_string(),
            function_type: DatapointKind::Json,
        })
        .await
        .unwrap();

    assert_eq!(
        new_count, 2,
        "Dataset should have 2 json datapoints after insertion"
    );
}

#[tokio::test]
async fn test_insert_datapoint_chat() {
    let clickhouse = get_clickhouse().await;

    let mut tags = HashMap::new();
    tags.insert("test".to_string(), "e2e".to_string());

    let new_datapoint_id = Uuid::now_v7();
    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: "test_insert_chat".to_string(),
        function_name: "write_haiku".to_string(),
        name: Some("test_chat_datapoint".to_string()),
        id: new_datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "response".to_string(),
        })]),
        tool_params: None,
        tags: Some(tags),
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    // Insert the datapoint
    clickhouse
        .insert_datapoints(&[datapoint_insert])
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Verify it was inserted by selecting
    let inserted_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: "test_insert_chat".to_string(),
            datapoint_id: new_datapoint_id,
            allow_stale: None,
        })
        .await;

    assert!(
        inserted_datapoint.is_ok(),
        "Should be able to select the inserted chat datapoint, but encountered error: {}",
        inserted_datapoint.unwrap_err()
    );
}

#[tokio::test]
async fn test_insert_datapoint_json() {
    let clickhouse = get_clickhouse().await;

    let mut tags = HashMap::new();
    tags.insert("test".to_string(), "e2e".to_string());

    let new_datapoint_id = Uuid::now_v7();
    let datapoint_insert = DatapointInsert::Json(JsonInferenceDatapointInsert {
        dataset_name: "test_insert_json".to_string(),
        function_name: "extract_entities".to_string(),
        name: Some("test_json_datapoint".to_string()),
        id: new_datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(JsonInferenceOutput {
            parsed: Some(json!({"data":"extracted"})),
            raw: Some("{\"data\":\"extracted\"}".to_string()),
        }),
        output_schema: json!({"type":"object"}),
        tags: Some(tags),
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    // Insert the datapoint
    clickhouse
        .insert_datapoints(&[datapoint_insert])
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Verify it was inserted by selecting
    let inserted_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: "test_insert_json".to_string(),
            datapoint_id: new_datapoint_id,
            allow_stale: None,
        })
        .await;
    assert!(
        inserted_datapoint.is_ok(),
        "Should be able to select the inserted json datapoint, but encountered error: {}",
        inserted_datapoint.unwrap_err()
    );
}

#[tokio::test]
async fn test_insert_datapoint_validates_dataset_name_builder() {
    let clickhouse = get_clickhouse().await;

    // Test reserved name "builder"
    let datapoint = ChatInferenceDatapointInsert {
        dataset_name: "builder".to_string(),
        function_name: "test_function".to_string(),
        id: Uuid::now_v7(),
        name: None,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "test".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    let result = clickhouse
        .insert_datapoints(&[DatapointInsert::Chat(datapoint)])
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_insert_datapoint_validates_dataset_name_tensorzero_prefix() {
    let clickhouse = get_clickhouse().await;

    // Test reserved prefix "tensorzero::"
    let datapoint = ChatInferenceDatapointInsert {
        dataset_name: "tensorzero::system".to_string(),
        function_name: "test_function".to_string(),
        id: Uuid::now_v7(),
        name: None,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "test".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    let result = clickhouse
        .insert_datapoints(&[DatapointInsert::Chat(datapoint)])
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_get_datapoint_returns_correct_json_datapoint_with_specific_id() {
    let clickhouse = get_clickhouse().await;
    let datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: "bar".to_string(),
            datapoint_id: Uuid::parse_str("01942e26-c48c-7720-b971-a1f7a3a9ac98").unwrap(),
            allow_stale: None,
        })
        .await
        .unwrap();

    if let StoredDatapoint::Json(datapoint) = datapoint {
        assert_eq!(datapoint.dataset_name, "bar");
        assert_eq!(datapoint.function_name, "ask_question");
        assert_eq!(
            datapoint.episode_id,
            Some(Uuid::parse_str("01942e26-4693-7e80-8591-47b98e25d721").unwrap())
        );
        assert_eq!(datapoint.name, None);
        assert_eq!(
            datapoint.id,
            Uuid::parse_str("01942e26-c48c-7720-b971-a1f7a3a9ac98").unwrap()
        );

        let input_messages = datapoint.input.messages;
        assert!(input_messages.contains(&StoredInputMessage {
            role: Role::User,
            content: vec![StoredInputMessageContent::Text(Text {
                text: "Is it a living thing?".to_string(),
            })],
        }));

        assert_eq!(
            datapoint.output.unwrap().parsed,
            Some(json!({
                "question": "Is it a large natural object, like a mountain or a tree?",
                "thinking": "Since the object is not a living thing and is not commonly found indoors, but is a natural object, it narrows down the possibilities to various elements from nature. It could be a rock, a tree, or potentially something like a mountain or a river. To further narrow it down, I will ask if it is a large object or a small object.",
            }))
        );

        assert_eq!(
            datapoint.output_schema,
            json!({
                "additionalProperties": false,
                "properties": {
                    "question": {
                        "type": "string",
                    },
                    "thinking": {
                        "type": "string",
                    },
                },
                "required": ["thinking", "question"],
                "type": "object"
            })
        );

        assert!(!datapoint.is_deleted, "is_deleted should be false");
        assert!(datapoint.staled_at.is_none(), "staled_at should be None");
        assert_eq!(datapoint.auxiliary, "", "auxiliary should be empty");
        assert_eq!(
            datapoint.source_inference_id, None,
            "source_inference_id should be None"
        );
        assert!(!datapoint.is_custom, "is_custom should be false");
    } else {
        panic!("Expected json datapoint");
    }
}

#[tokio::test]
async fn test_get_datapoint_returns_correct_chat_datapoint_with_specific_id() {
    let clickhouse = get_clickhouse().await;
    let datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: "foo".to_string(),
            datapoint_id: Uuid::parse_str("01934fc5-ea98-71f0-8191-9fd88f34c28b").unwrap(),
            allow_stale: None,
        })
        .await
        .unwrap();

    if let StoredDatapoint::Chat(datapoint) = datapoint {
        assert_eq!(datapoint.dataset_name, "foo");
        assert_eq!(datapoint.function_name, "write_haiku");
        assert_eq!(
            datapoint.episode_id,
            Some(Uuid::parse_str("0193fb9d-73ad-7ad2-807d-a2ef10088ff9").unwrap())
        );
        assert_eq!(datapoint.name, None);
        assert_eq!(
            datapoint.id,
            Uuid::parse_str("01934fc5-ea98-71f0-8191-9fd88f34c28b").unwrap()
        );

        assert!(!datapoint.is_deleted, "is_deleted should be false");
        assert_eq!(datapoint.auxiliary, "", "auxiliary should be empty");
        assert!(datapoint.staled_at.is_none(), "staled_at should be None");
        assert_eq!(
            datapoint.source_inference_id, None,
            "source_inference_id should be None"
        );
        assert!(!datapoint.is_custom, "is_custom should be false");
    } else {
        panic!("Expected chat datapoint");
    }
}

#[tokio::test]
async fn test_get_datapoint_returns_error_for_non_existent_datapoint() {
    let clickhouse = get_clickhouse().await;
    let result = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: "foo".to_string(),
            datapoint_id: Uuid::parse_str("00000000-0000-0000-0000-000000000000").unwrap(),
            allow_stale: None,
        })
        .await;

    assert!(
        result.is_err(),
        "Should return error for non-existent datapoint"
    );
}

#[tokio::test]
async fn test_chat_datapoint_lifecycle_insert_get_delete() {
    let clickhouse = get_clickhouse().await;
    let datapoint_id = Uuid::now_v7();
    let source_inference_id = Uuid::now_v7();

    let mut tags = HashMap::new();
    tags.insert("test".to_string(), "lifecycle".to_string());

    let chat_datapoint = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: "test_chat_dataset".to_string(),
        function_name: "write_haiku".to_string(),
        id: datapoint_id,
        episode_id: Some(Uuid::parse_str("0193fb9d-73ad-7ad2-807d-a2ef10088ff9").unwrap()),
        name: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Code flows like water\nTests catch bugs in their net now\nPeace in the program"
                .to_string(),
        })]),
        tool_params: None,
        tags: Some(tags),
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: Some(source_inference_id),
        is_custom: false,
    });

    // Test insertion
    clickhouse
        .insert_datapoints(&[chat_datapoint])
        .await
        .unwrap();

    // Flush async insert to ensure datapoint is visible before deletion
    clickhouse_flush_async_insert(&clickhouse).await;

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Test retrieval
    let retrieved_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: "test_chat_dataset".to_string(),
            datapoint_id,
            allow_stale: None,
        })
        .await
        .unwrap();

    assert_eq!(retrieved_datapoint.id(), datapoint_id);
    assert_eq!(retrieved_datapoint.function_name(), "write_haiku");
    assert_eq!(retrieved_datapoint.dataset_name(), "test_chat_dataset");

    if let StoredDatapoint::Chat(chat_dp) = retrieved_datapoint {
        assert_eq!(chat_dp.source_inference_id, Some(source_inference_id));
    } else {
        panic!("Expected chat datapoint");
    }

    // Test staling
    clickhouse
        .delete_datapoints("test_chat_dataset", Some(&[datapoint_id]))
        .await
        .unwrap();

    // Flush async insert to ensure datapoint is deleted
    clickhouse_flush_async_insert(&clickhouse).await;

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Try to get the datapoint (should return error since it's staled)
    let staled_result = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: "test_chat_dataset".to_string(),
            datapoint_id,
            allow_stale: None,
        })
        .await;

    assert!(
        staled_result.is_err(),
        "Should return error for staled datapoint"
    );

    // Test retrieval with allow_stale=true
    let staled_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: "test_chat_dataset".to_string(),
            datapoint_id,
            allow_stale: Some(true),
        })
        .await
        .unwrap();

    assert_eq!(staled_datapoint.id(), datapoint_id);

    if let StoredDatapoint::Chat(chat_dp) = staled_datapoint {
        assert!(
            chat_dp.staled_at.is_some(),
            "Should have staled_at timestamp"
        );
    } else {
        panic!("Expected chat datapoint");
    }
}

#[tokio::test]
async fn test_json_datapoint_lifecycle_insert_get_delete() {
    let clickhouse = get_clickhouse().await;
    let datapoint_id = Uuid::now_v7();
    let source_inference_id = Uuid::now_v7();

    let mut tags = HashMap::new();
    tags.insert("test".to_string(), "lifecycle".to_string());

    let json_datapoint = DatapointInsert::Json(JsonInferenceDatapointInsert {
        dataset_name: "test_json_dataset".to_string(),
        function_name: "extract_entities".to_string(),
        id: datapoint_id,
        episode_id: Some(Uuid::parse_str("0193fb9d-73ad-7ad2-807d-a2ef10088ff8").unwrap()),
        name: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: None,
        output_schema: json!({"type":"object"}),
        tags: Some(tags),
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: Some(source_inference_id),
        is_custom: false,
    });

    // Test insertion
    clickhouse
        .insert_datapoints(&[json_datapoint])
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Test retrieval
    let retrieved_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: "test_json_dataset".to_string(),
            datapoint_id,
            allow_stale: None,
        })
        .await
        .unwrap();

    assert_eq!(retrieved_datapoint.id(), datapoint_id);
    assert_eq!(retrieved_datapoint.function_name(), "extract_entities");
    assert_eq!(retrieved_datapoint.dataset_name(), "test_json_dataset");

    if let StoredDatapoint::Json(json_dp) = retrieved_datapoint {
        assert_eq!(json_dp.source_inference_id, Some(source_inference_id));
    } else {
        panic!("Expected json datapoint");
    }

    // Test staling
    clickhouse
        .delete_datapoints("test_json_dataset", Some(&[datapoint_id]))
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Try to get the datapoint (should return error since it's staled)
    let staled_result = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: "test_json_dataset".to_string(),
            datapoint_id,
            allow_stale: None,
        })
        .await;

    assert!(
        staled_result.is_err(),
        "Should return error for staled datapoint"
    );

    // Test retrieval with allow_stale=true
    let staled_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: "test_json_dataset".to_string(),
            datapoint_id,
            allow_stale: Some(true),
        })
        .await
        .unwrap();

    assert_eq!(staled_datapoint.id(), datapoint_id);

    if let StoredDatapoint::Json(json_dp) = staled_datapoint {
        assert!(
            json_dp.staled_at.is_some(),
            "Should have staled_at timestamp"
        );
    } else {
        panic!("Expected json datapoint");
    }
}

#[tokio::test]
async fn test_handles_non_existent_datapoint_retrieval() {
    let clickhouse = get_clickhouse().await;
    let result = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: "non_existent_dataset".to_string(),
            datapoint_id: Uuid::parse_str("01934fc5-ea98-71f0-8191-9fd88f34c30d").unwrap(),
            allow_stale: None,
        })
        .await;

    assert!(
        result.is_err(),
        "Should return error for non-existent datapoint"
    );
}

#[tokio::test]
async fn test_handles_duplicate_insertions_gracefully() {
    let clickhouse = get_clickhouse().await;
    let datapoint_id = Uuid::parse_str("01934fc5-ea98-71f0-8191-9fd88f34c31e").unwrap();
    let source_inference_id = Uuid::now_v7();

    let mut tags = HashMap::new();
    tags.insert("test".to_string(), "duplicate".to_string());

    let chat_datapoint = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: "test_chat_dataset".to_string(),
        function_name: "write_haiku".to_string(),
        id: datapoint_id,
        episode_id: Some(Uuid::parse_str("0193fb9d-73ad-7ad2-807d-a2ef10088ff7").unwrap()),
        name: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Copies everywhere\nDuplicates fill the database\nUnique keys break down"
                .to_string(),
        })]),
        tool_params: None,
        tags: Some(tags),
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: Some(source_inference_id),
        is_custom: false,
    });
    let datapoint_slice = [chat_datapoint];

    // First insertion
    clickhouse
        .insert_datapoints(&datapoint_slice)
        .await
        .unwrap();

    // Second insertion with same ID should not throw
    clickhouse
        .insert_datapoints(&datapoint_slice)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_handles_staling_of_non_existent_datapoint() {
    let clickhouse = get_clickhouse().await;

    // Should not throw when trying to stale a non-existent datapoint
    let datapoint_id = Uuid::now_v7();
    let result = clickhouse
        .delete_datapoints("fake", Some(&[datapoint_id]))
        .await;

    // This should succeed without error (graceful handling)
    assert!(
        result.is_ok(),
        "Should handle staling non-existent datapoint gracefully, but encountered error: {}",
        result.unwrap_err()
    );
}

#[tokio::test]
async fn test_count_datapoints_for_dataset_function_chat_write_haiku() {
    let clickhouse = get_clickhouse().await;
    let count = clickhouse
        .count_datapoints_for_dataset_function(&CountDatapointsForDatasetFunctionParams {
            dataset_name: "foo".to_string(),
            function_name: "write_haiku".to_string(),
            function_type: DatapointKind::Chat,
        })
        .await
        .unwrap();

    // Based on existing test data, we expect some chat datapoints for write_haiku function in foo dataset
    // TODO(#3903): Stop making assumptions about what data exists in the database.
    assert!(count > 0, "Should have some chat datapoints");
}

#[tokio::test]
async fn test_count_datapoints_for_dataset_function_json_extract_entities() {
    let clickhouse = get_clickhouse().await;
    let count = clickhouse
        .count_datapoints_for_dataset_function(&CountDatapointsForDatasetFunctionParams {
            dataset_name: "foo".to_string(),
            function_name: "extract_entities".to_string(),
            function_type: DatapointKind::Json,
        })
        .await
        .unwrap();

    // Based on existing test data, we expect some json datapoints for extract_entities function in foo dataset
    // TODO(#3903): Stop making assumptions about what data exists in the database.
    assert_eq!(count, 43, "Should have 43 json datapoints");
}

#[tokio::test]
async fn test_count_datapoints_for_dataset_function_non_existent_dataset() {
    let clickhouse = get_clickhouse().await;
    let count = clickhouse
        .count_datapoints_for_dataset_function(&CountDatapointsForDatasetFunctionParams {
            dataset_name: "fake".to_string(),
            function_name: "write_haiku".to_string(),
            function_type: DatapointKind::Chat,
        })
        .await
        .unwrap();

    assert_eq!(
        count, 0,
        "Should have 0 datapoints for non-existent dataset"
    );
}

#[tokio::test]
async fn test_count_datapoints_for_dataset_function_non_existent_function() {
    let clickhouse = get_clickhouse().await;
    let count = clickhouse
        .count_datapoints_for_dataset_function(&CountDatapointsForDatasetFunctionParams {
            dataset_name: "foo".to_string(),
            function_name: "fake".to_string(),
            function_type: DatapointKind::Chat,
        })
        .await
        .unwrap();

    assert_eq!(
        count, 0,
        "Should have 0 datapoints for non-existent function"
    );
}

#[tokio::test]
async fn test_insert_rows_for_dataset_handles_invalid_dataset_names() {
    let clickhouse = get_clickhouse().await;

    let params = DatasetQueryParams {
        inference_type: DatapointKind::Chat,
        function_name: None,
        dataset_name: Some("builder".to_string()),
        variant_name: None,
        extra_where: None,
        extra_params: None,
        metric_filter: None,
        output_source: DatasetOutputSource::None,
        limit: None,
        offset: None,
    };

    let result = clickhouse.insert_rows_for_dataset(&params).await;
    assert!(
        result.is_err(),
        "Should reject reserved dataset name 'builder'"
    );
}

#[tokio::test]
async fn test_insert_datapoint_handles_invalid_dataset_names() {
    let clickhouse = get_clickhouse().await;
    let mut tags = HashMap::new();
    tags.insert("test".to_string(), "invalid_name".to_string());

    let chat_datapoint = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: "builder".to_string(),
        function_name: "write_haiku".to_string(),
        id: Uuid::now_v7(),
        episode_id: None,
        name: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![]),
        tool_params: None,
        tags: Some(tags),
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    let result = clickhouse.insert_datapoints(&[chat_datapoint]).await;
    assert!(
        result.is_err(),
        "Should reject reserved dataset name 'builder'"
    );
}

#[tokio::test]
async fn test_get_datapoints_with_empty_ids() {
    let clickhouse = get_clickhouse().await;

    let result = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some("test_dataset".to_string()),
            function_name: None,
            ids: None,
            limit: 20,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();

    assert_eq!(result.len(), 0, "Should return empty vector for empty IDs");
}

#[tokio::test]
async fn test_get_datapoints_with_single_chat_datapoint() {
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test_get_datapoints_{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    // Insert a chat datapoint
    let datapoint = ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_function".to_string(),
        id: datapoint_id,
        name: Some("test_chat".to_string()),
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "test response".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    clickhouse
        .insert_datapoints(&[DatapointInsert::Chat(datapoint)])
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve using get_datapoints
    let result = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: Some(vec![datapoint_id]),
            limit: 20,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();

    assert_eq!(result.len(), 1, "Should return exactly one datapoint");

    if let StoredDatapoint::Chat(dp) = &result[0] {
        assert_eq!(dp.id, datapoint_id);
        assert_eq!(dp.function_name, "test_function");
        assert_eq!(dp.name, Some("test_chat".to_string()));
    } else {
        panic!("Expected chat datapoint");
    }
}

#[tokio::test]
async fn test_get_datapoints_with_single_json_datapoint() {
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test_get_datapoints_{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    // Insert a json datapoint
    let datapoint = JsonInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_function".to_string(),
        id: datapoint_id,
        name: Some("test_json".to_string()),
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(JsonInferenceOutput {
            parsed: Some(json!({"key": "value"})),
            raw: Some("{\"key\":\"value\"}".to_string()),
        }),
        output_schema: json!({"type": "object"}),
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    clickhouse
        .insert_datapoints(&[DatapointInsert::Json(datapoint)])
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve using get_datapoints
    let result = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: Some(vec![datapoint_id]),
            limit: 20,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();

    assert_eq!(result.len(), 1, "Should return exactly one datapoint");

    if let StoredDatapoint::Json(dp) = &result[0] {
        assert_eq!(dp.id, datapoint_id);
        assert_eq!(dp.function_name, "test_function");
        assert_eq!(dp.name, Some("test_json".to_string()));
    } else {
        panic!("Expected json datapoint");
    }
}

#[tokio::test]
async fn test_get_datapoints_with_multiple_mixed_datapoints() {
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test_get_datapoints_{}", Uuid::now_v7());

    // Create IDs
    let chat_id1 = Uuid::now_v7();
    let json_id = Uuid::now_v7();
    let chat_id2 = Uuid::now_v7();

    // Insert chat datapoint 1
    let chat_dp1 = ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_chat_function".to_string(),
        id: chat_id1,
        name: Some("chat1".to_string()),
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "chat response 1".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    clickhouse
        .insert_datapoints(&[DatapointInsert::Chat(chat_dp1)])
        .await
        .unwrap();

    // Insert json datapoint
    let json_dp = JsonInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_json_function".to_string(),
        id: json_id,
        name: Some("json1".to_string()),
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(JsonInferenceOutput {
            parsed: Some(json!({"data": "test"})),
            raw: Some("{\"data\":\"test\"}".to_string()),
        }),
        output_schema: json!({"type": "object"}),
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    clickhouse
        .insert_datapoints(&[DatapointInsert::Json(json_dp)])
        .await
        .unwrap();

    // Insert chat datapoint 2
    let chat_dp2 = ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_chat_function".to_string(),
        id: chat_id2,
        name: Some("chat2".to_string()),
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "chat response 2".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    clickhouse
        .insert_datapoints(&[DatapointInsert::Chat(chat_dp2)])
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve all three datapoints
    let result = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: Some(vec![chat_id1, json_id, chat_id2]),
            limit: 20,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();

    assert_eq!(
        result.len(),
        3,
        "Should return all three datapoints (2 chat, 1 json)"
    );

    // Verify we got all the expected IDs
    let returned_ids: Vec<Uuid> = result.iter().map(StoredDatapoint::id).collect();
    assert!(returned_ids.contains(&chat_id1));
    assert!(returned_ids.contains(&json_id));
    assert!(returned_ids.contains(&chat_id2));

    // Count types
    let chat_count = result
        .iter()
        .filter(|dp| matches!(dp, StoredDatapoint::Chat(_)))
        .count();
    let json_count = result
        .iter()
        .filter(|dp| matches!(dp, StoredDatapoint::Json(_)))
        .count();

    assert_eq!(chat_count, 2, "Should have 2 chat datapoints");
    assert_eq!(json_count, 1, "Should have 1 json datapoint");
}

#[tokio::test]
async fn test_get_datapoints_with_non_existent_ids() {
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test_get_datapoints_{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    // Insert one datapoint
    let datapoint = ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_function".to_string(),
        id: datapoint_id,
        name: None,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "test".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    clickhouse
        .insert_datapoints(&[DatapointInsert::Chat(datapoint)])
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Query with both existing and non-existent IDs
    let non_existent_id = Uuid::now_v7();
    let another_non_existent_id = Uuid::now_v7();
    let result = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: Some(vec![datapoint_id, non_existent_id, another_non_existent_id]),
            limit: 20,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();

    assert_eq!(
        result.len(),
        1,
        "Should only return the one existing datapoint"
    );
    assert_eq!(result[0].id(), datapoint_id);
}

#[tokio::test]
async fn test_get_datapoints_with_search_query() {
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test_get_datapoints_{}", Uuid::now_v7());

    // Create IDs
    let chat_id1 = Uuid::now_v7();
    let json_id = Uuid::now_v7();
    let chat_id2 = Uuid::now_v7();

    // Insert chat datapoint 1
    let chat_dp1 = ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_chat_function".to_string(),
        id: chat_id1,
        name: Some("chat1".to_string()),
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "chat response 1".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    clickhouse
        .insert_datapoints(&[DatapointInsert::Chat(chat_dp1)])
        .await
        .unwrap();

    // Insert json datapoint
    let json_dp = JsonInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_json_function".to_string(),
        id: json_id,
        name: Some("json1".to_string()),
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(JsonInferenceOutput {
            parsed: Some(json!({"data": "test"})),
            raw: Some("{\"data\":\"test\"}".to_string()),
        }),
        output_schema: json!({"type": "object"}),
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    clickhouse
        .insert_datapoints(&[DatapointInsert::Json(json_dp)])
        .await
        .unwrap();

    // Insert chat datapoint 2
    let chat_dp2 = ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_chat_function".to_string(),
        id: chat_id2,
        name: Some("chat2".to_string()),
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "chat response 2".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    clickhouse
        .insert_datapoints(&[DatapointInsert::Chat(chat_dp2)])
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve all three datapoints
    let result = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: None,
            limit: 20,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: Some(vec![DatapointOrderBy {
                term: DatapointOrderByTerm::SearchRelevance,
                direction: OrderDirection::Desc,
            }]),
            search_query_experimental: Some("chat".to_string()),
        })
        .await
        .unwrap();

    assert_eq!(result.len(), 2, "Should return 2 datapoints (both chats)");

    // Verify we got all the expected IDs
    let returned_ids: HashSet<Uuid> = result.iter().map(StoredDatapoint::id).collect();
    assert!(returned_ids.contains(&chat_id1));
    assert!(returned_ids.contains(&chat_id2));
}

#[tokio::test]
async fn test_get_datapoints_with_search_query_with_json_encoded_term() {
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test_get_datapoints_{}", Uuid::now_v7());

    // Insert json datapoint with escaped content
    let json_id = Uuid::now_v7();
    let parsed_value = json!({"data": "this is an input string with \"escaped\" content"});
    let json_dp = JsonInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_json_function".to_string(),
        id: json_id,
        name: Some("json1".to_string()),
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(JsonInferenceOutput {
            parsed: Some(parsed_value.clone()),
            raw: Some(serde_json::to_string(&parsed_value).unwrap()),
        }),
        output_schema: json!({"type": "object"}),
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    clickhouse
        .insert_datapoints(&[DatapointInsert::Json(json_dp)])
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve all three datapoints
    let result = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: None,
            limit: 20,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: Some(vec![DatapointOrderBy {
                term: DatapointOrderByTerm::SearchRelevance,
                direction: OrderDirection::Desc,
            }]),
            search_query_experimental: Some(r#""escaped" content"#.to_string()),
        })
        .await
        .unwrap();

    assert_eq!(
        result.len(),
        1,
        "Should return the newly inserted json datapoint"
    );
    assert_eq!(result[0].id(), json_id);
}

#[tokio::test]
async fn test_get_datapoints_respects_allow_stale_false() {
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test_get_datapoints_{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    // Insert a datapoint
    let datapoint = ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_function".to_string(),
        id: datapoint_id,
        name: None,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "test".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    clickhouse
        .insert_datapoints(&[DatapointInsert::Chat(datapoint)])
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Verify we can retrieve it before staling
    let result = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: Some(vec![datapoint_id]),
            limit: 20,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();
    assert_eq!(result.len(), 1, "Should retrieve datapoint before staling");

    // Stale the datapoint
    clickhouse
        .delete_datapoints(&dataset_name, Some(&[datapoint_id]))
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Try to retrieve with allow_stale=false
    let result = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: Some(vec![datapoint_id]),
            limit: 20,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();

    assert_eq!(
        result.len(),
        0,
        "Should not return staled datapoint when allow_stale=false"
    );
}

#[tokio::test]
async fn test_get_datapoints_respects_allow_stale_true() {
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test_get_datapoints_{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    // Insert a datapoint
    let datapoint = ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_function".to_string(),
        id: datapoint_id,
        name: None,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "test".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    clickhouse
        .insert_datapoints(&[DatapointInsert::Chat(datapoint)])
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Stale the datapoint
    clickhouse
        .delete_datapoints(&dataset_name, Some(&[datapoint_id]))
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Try to retrieve with allow_stale=true
    let result = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(dataset_name.clone()),
            function_name: None,
            ids: Some(vec![datapoint_id]),
            limit: 20,
            offset: 0,
            allow_stale: true,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();

    assert_eq!(
        result.len(),
        1,
        "Should return staled datapoint when allow_stale=true"
    );

    if let StoredDatapoint::Chat(dp) = &result[0] {
        assert!(
            dp.staled_at.is_some(),
            "Datapoint should have staled_at timestamp"
        );
    } else {
        panic!("Expected chat datapoint");
    }
}

#[tokio::test]
async fn test_get_datapoints_with_wrong_dataset_name() {
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("test_get_datapoints_{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    // Insert a datapoint in one dataset
    let datapoint = ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_function".to_string(),
        id: datapoint_id,
        name: None,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "test".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    };

    clickhouse
        .insert_datapoints(&[DatapointInsert::Chat(datapoint)])
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Try to retrieve with different dataset name
    let wrong_dataset = format!("wrong_{dataset_name}");
    let result = clickhouse
        .get_datapoints(&GetDatapointsParams {
            dataset_name: Some(wrong_dataset),
            function_name: None,
            ids: Some(vec![datapoint_id]),
            limit: 20,
            offset: 0,
            allow_stale: false,
            filter: None,
            order_by: None,
            search_query_experimental: None,
        })
        .await
        .unwrap();

    assert_eq!(
        result.len(),
        0,
        "Should not return datapoint when querying wrong dataset"
    );
}

#[tokio::test]
async fn test_chat_datapoint_with_file_object_storage_roundtrip() {
    let clickhouse = get_clickhouse().await;
    let datapoint_id = Uuid::now_v7();
    let dataset_name = format!("test_file_storage_{}", Uuid::now_v7());

    // Create a StoredFile with ObjectStorage
    let stored_file = StoredFile(ObjectStoragePointer {
        source_url: Some("https://example.com/original.png".parse().unwrap()),
        detail: None,
        mime_type: mime::IMAGE_PNG,
        storage_path: StoragePath {
            kind: StorageKind::Disabled,
            path: ObjectStorePath::parse("test/files/image.png").unwrap(),
        },
        filename: None,
    });

    let chat_datapoint = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_function".to_string(),
        id: datapoint_id,
        name: Some("test_with_file".to_string()),
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::File(Box::new(
                    stored_file.clone(),
                ))],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "response".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    // Insert the datapoint
    clickhouse
        .insert_datapoints(&[chat_datapoint])
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve the datapoint
    let retrieved_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id,
            allow_stale: None,
        })
        .await
        .unwrap();

    // Verify the file was preserved correctly
    if let StoredDatapoint::Chat(chat_dp) = retrieved_datapoint {
        assert_eq!(chat_dp.id, datapoint_id);
        assert_eq!(chat_dp.input.messages.len(), 1);
        assert_eq!(chat_dp.input.messages[0].content.len(), 1);

        match &chat_dp.input.messages[0].content[0] {
            StoredInputMessageContent::File(file) => {
                assert_eq!(file.mime_type, mime::IMAGE_PNG);
                assert_eq!(
                    file.source_url,
                    Some("https://example.com/original.png".parse().unwrap())
                );
                assert_eq!(file.storage_path.path, stored_file.storage_path.path);
            }
            _ => panic!("Expected File content"),
        }
    } else {
        panic!("Expected chat datapoint");
    }
}

#[tokio::test]
async fn test_json_datapoint_with_file_object_storage_roundtrip() {
    let clickhouse = get_clickhouse().await;
    let datapoint_id = Uuid::now_v7();
    let dataset_name = format!("test_file_storage_{}", Uuid::now_v7());

    // Create a StoredFile with ObjectStorage
    let stored_file = StoredFile(ObjectStoragePointer {
        source_url: Some("https://example.com/data.json".parse().unwrap()),
        detail: None,
        mime_type: mime::APPLICATION_JSON,
        storage_path: StoragePath {
            kind: StorageKind::Disabled,
            path: ObjectStorePath::parse("test/files/data.json").unwrap(),
        },
        filename: None,
    });

    let json_datapoint = DatapointInsert::Json(JsonInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_function".to_string(),
        id: datapoint_id,
        name: Some("test_json_with_file".to_string()),
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::File(Box::new(
                    stored_file.clone(),
                ))],
            }],
        },
        output: Some(JsonInferenceOutput {
            parsed: Some(json!({"result": "success"})),
            raw: Some("{\"result\":\"success\"}".to_string()),
        }),
        output_schema: json!({"type": "object"}),
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    // Insert the datapoint
    clickhouse
        .insert_datapoints(&[json_datapoint])
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve the datapoint
    let retrieved_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id,
            allow_stale: None,
        })
        .await
        .unwrap();

    // Verify the file was preserved correctly
    if let StoredDatapoint::Json(json_dp) = retrieved_datapoint {
        assert_eq!(json_dp.id, datapoint_id);
        assert_eq!(json_dp.input.messages.len(), 1);
        assert_eq!(json_dp.input.messages[0].content.len(), 1);

        match &json_dp.input.messages[0].content[0] {
            StoredInputMessageContent::File(file) => {
                assert_eq!(file.mime_type, mime::APPLICATION_JSON);
                assert_eq!(
                    file.source_url,
                    Some("https://example.com/data.json".parse().unwrap())
                );
                assert_eq!(file.storage_path.path, stored_file.storage_path.path);
            }
            _ => panic!("Expected File content"),
        }
    } else {
        panic!("Expected json datapoint");
    }
}

#[tokio::test]
async fn test_datapoint_with_mixed_file_types() {
    let clickhouse = get_clickhouse().await;
    let datapoint_id = Uuid::now_v7();
    let dataset_name = format!("test_mixed_files_{}", Uuid::now_v7());

    // Create multiple StoredFiles
    let stored_file1 = StoredFile(ObjectStoragePointer {
        source_url: Some("https://example.com/image1.png".parse().unwrap()),
        detail: None,
        mime_type: mime::IMAGE_PNG,
        storage_path: StoragePath {
            kind: StorageKind::Disabled,
            path: ObjectStorePath::parse("test/files/image1.png").unwrap(),
        },
        filename: None,
    });

    let stored_file2 = StoredFile(ObjectStoragePointer {
        source_url: None, // No source URL
        detail: None,
        mime_type: mime::IMAGE_JPEG,
        storage_path: StoragePath {
            kind: StorageKind::Disabled,
            path: ObjectStorePath::parse("test/files/image2.jpg").unwrap(),
        },
        filename: None,
    });

    let chat_datapoint = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_function".to_string(),
        id: datapoint_id,
        name: Some("test_mixed_files".to_string()),
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![
                StoredInputMessage {
                    role: Role::User,
                    content: vec![
                        StoredInputMessageContent::Text(Text {
                            text: "Here are some files".to_string(),
                        }),
                        StoredInputMessageContent::File(Box::new(stored_file1.clone())),
                    ],
                },
                StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::File(Box::new(
                        stored_file2.clone(),
                    ))],
                },
            ],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "response".to_string(),
        })]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    // Insert the datapoint
    clickhouse
        .insert_datapoints(&[chat_datapoint])
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve the datapoint
    let retrieved_datapoint = clickhouse
        .get_datapoint(&GetDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id,
            allow_stale: None,
        })
        .await
        .unwrap();

    // Verify all files were preserved correctly
    if let StoredDatapoint::Chat(chat_dp) = retrieved_datapoint {
        assert_eq!(chat_dp.id, datapoint_id);
        assert_eq!(chat_dp.input.messages.len(), 2);

        // Check first message with text and file
        assert_eq!(chat_dp.input.messages[0].content.len(), 2);
        match &chat_dp.input.messages[0].content[0] {
            StoredInputMessageContent::Text(text) => {
                assert_eq!(text.text, "Here are some files");
            }
            _ => panic!("Expected Text content"),
        }
        match &chat_dp.input.messages[0].content[1] {
            StoredInputMessageContent::File(file) => {
                assert_eq!(file.mime_type, mime::IMAGE_PNG);
                assert_eq!(
                    file.source_url,
                    Some("https://example.com/image1.png".parse().unwrap())
                );
                assert_eq!(file.storage_path.path, stored_file1.storage_path.path);
            }
            _ => panic!("Expected File content"),
        }

        // Check second message with file only
        assert_eq!(chat_dp.input.messages[1].content.len(), 1);
        match &chat_dp.input.messages[1].content[0] {
            StoredInputMessageContent::File(file) => {
                assert_eq!(file.mime_type, mime::IMAGE_JPEG);
                assert_eq!(file.source_url, None);
                assert_eq!(file.storage_path.path, stored_file2.storage_path.path);
            }
            _ => panic!("Expected File content"),
        }
    } else {
        panic!("Expected chat datapoint");
    }
}

// Tool Call Storage Format Tests (Migration 0041)
// These tests verify the new decomposed storage format for tool calls

mod tool_call_storage_tests {
    use super::*;
    use serde_json::json;
    use tensorzero_core::tool::{
        AllowedTools, AllowedToolsChoice, FunctionTool, ProviderTool, ProviderToolScope,
        ProviderToolScopeModelProvider, Tool, ToolCallConfigDatabaseInsert, ToolChoice,
    };

    #[tokio::test]
    async fn test_tool_call_storage_static_tools_only() {
        // Test Case 1: Static tools only (from function config, not provided dynamically)
        let clickhouse = get_clickhouse().await;
        let datapoint_id = Uuid::now_v7();
        let dataset_name = format!("test_tool_storage_{}", Uuid::now_v7());

        let datapoint = ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "test_function".to_string(),
            id: datapoint_id,
            name: None,
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "response".to_string(),
            })]),
            tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
                vec![], // No dynamic tools
                vec![], // No provider tools
                AllowedTools {
                    tools: ["static_tool_1".to_string(), "static_tool_2".to_string()]
                        .into_iter()
                        .collect(),
                    choice: AllowedToolsChoice::Explicit,
                },
                ToolChoice::Auto,
                None,
            )),
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };

        clickhouse
            .insert_datapoints(&[DatapointInsert::Chat(datapoint)])
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        // Verify by retrieving the datapoint
        let retrieved_datapoint = clickhouse
            .get_datapoint(&GetDatapointParams {
                dataset_name: dataset_name.clone(),
                datapoint_id,
                allow_stale: None,
            })
            .await
            .unwrap();

        if let StoredDatapoint::Chat(chat_dp) = retrieved_datapoint {
            // Verify roundtrip - tool_params should be reconstructed correctly
            assert!(chat_dp.tool_params.is_some());
            let tool_params = chat_dp.tool_params.unwrap();

            // No dynamic tools
            assert!(tool_params.dynamic_tools.is_empty());

            // No provider tools
            assert!(tool_params.dynamic_provider_tools.is_empty());

            // Allowed tools should contain static tools
            assert_eq!(tool_params.allowed_tools.tools.len(), 2);
            assert!(tool_params
                .allowed_tools
                .tools
                .contains(&"static_tool_1".to_string()));
            assert!(tool_params
                .allowed_tools
                .tools
                .contains(&"static_tool_2".to_string()));
            assert_eq!(
                tool_params.allowed_tools.choice,
                AllowedToolsChoice::Explicit
            );

            assert_eq!(tool_params.tool_choice, ToolChoice::Auto);
            assert_eq!(tool_params.parallel_tool_calls, None);
        } else {
            panic!("Expected chat datapoint");
        }
    }

    #[tokio::test]
    async fn test_tool_call_storage_dynamic_tools_only() {
        // Test Case 2: Dynamic tools only (provided at runtime)
        let clickhouse = get_clickhouse().await;
        let datapoint_id = Uuid::now_v7();
        let dataset_name = format!("test_tool_storage_{}", Uuid::now_v7());

        let dynamic_tool = Tool::Function(FunctionTool {
            name: "runtime_tool".to_string(),
            description: "A tool provided at runtime".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
            strict: false,
        });

        let datapoint = ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "test_function".to_string(),
            id: datapoint_id,
            name: None,
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "response".to_string(),
            })]),
            tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
                vec![dynamic_tool], // Dynamic tool
                vec![],             // No provider tools
                AllowedTools {
                    tools: vec![], // Empty static tools
                    choice: AllowedToolsChoice::Explicit,
                },
                ToolChoice::Required,
                Some(true),
            )),
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };

        clickhouse
            .insert_datapoints(&[DatapointInsert::Chat(datapoint)])
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        let retrieved_datapoint = clickhouse
            .get_datapoint(&GetDatapointParams {
                dataset_name: dataset_name.clone(),
                datapoint_id,
                allow_stale: None,
            })
            .await
            .unwrap();

        if let StoredDatapoint::Chat(chat_dp) = retrieved_datapoint {
            assert!(chat_dp.tool_params.is_some());
            let tool_params = chat_dp.tool_params.unwrap();

            // Verify dynamic tool is present
            assert_eq!(tool_params.dynamic_tools.len(), 1);

            let Tool::Function(tool) = &tool_params.dynamic_tools[0] else {
                panic!("Expected Function tool");
            };
            assert_eq!(tool.name, "runtime_tool");
            assert_eq!(tool.description, "A tool provided at runtime");
            assert!(!tool.strict);

            // No static tools (empty allowed_tools)
            assert!(tool_params.allowed_tools.tools.is_empty());
            assert_eq!(
                tool_params.allowed_tools.choice,
                AllowedToolsChoice::Explicit
            );

            assert_eq!(tool_params.tool_choice, ToolChoice::Required);
            assert_eq!(tool_params.parallel_tool_calls, Some(true));
        } else {
            panic!("Expected chat datapoint");
        }
    }

    #[tokio::test]
    async fn test_tool_call_storage_mixed_static_and_dynamic() {
        // Test Case 3: Mixed static + dynamic tools
        let clickhouse = get_clickhouse().await;
        let datapoint_id = Uuid::now_v7();
        let dataset_name = format!("test_tool_storage_{}", Uuid::now_v7());

        let dynamic_tool = Tool::Function(FunctionTool {
            name: "dynamic_x".to_string(),
            description: "Dynamic tool X".to_string(),
            parameters: json!({"type": "object"}),
            strict: true,
        });

        let datapoint = ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "test_function".to_string(),
            id: datapoint_id,
            name: None,
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "response".to_string(),
            })]),
            tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
                vec![dynamic_tool],
                vec![],
                AllowedTools {
                    tools: vec!["static_a".to_string(), "static_b".to_string()],
                    choice: AllowedToolsChoice::Explicit,
                },
                ToolChoice::Auto,
                None,
            )),
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };

        clickhouse
            .insert_datapoints(&[DatapointInsert::Chat(datapoint)])
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        let retrieved_datapoint = clickhouse
            .get_datapoint(&GetDatapointParams {
                dataset_name: dataset_name.clone(),
                datapoint_id,
                allow_stale: None,
            })
            .await
            .unwrap();

        if let StoredDatapoint::Chat(chat_dp) = retrieved_datapoint {
            assert!(chat_dp.tool_params.is_some());
            let tool_params = chat_dp.tool_params.unwrap();

            // Verify both static and dynamic tools
            assert_eq!(tool_params.dynamic_tools.len(), 1);
            assert_eq!(tool_params.allowed_tools.tools.len(), 2);
            assert!(tool_params
                .allowed_tools
                .tools
                .contains(&"static_a".to_string()));
            assert!(tool_params
                .allowed_tools
                .tools
                .contains(&"static_b".to_string()));
        } else {
            panic!("Expected chat datapoint");
        }
    }

    #[tokio::test]
    async fn test_tool_call_storage_provider_tools() {
        // Test Case 4: Provider tools (model-provider-specific tools)
        let clickhouse = get_clickhouse().await;
        let datapoint_id = Uuid::now_v7();
        let dataset_name = format!("test_tool_storage_{}", Uuid::now_v7());

        let provider_tool = ProviderTool {
            scope: ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                model_name: "gpt-4".to_string(),
                provider_name: Some("openai".to_string()),
            }),
            tool: json!({
                "type": "code_interpreter"
            }),
        };

        let datapoint = ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "test_function".to_string(),
            id: datapoint_id,
            name: None,
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "response".to_string(),
            })]),
            tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
                vec![],
                vec![provider_tool],
                AllowedTools {
                    tools: vec![],
                    choice: AllowedToolsChoice::FunctionDefault,
                },
                ToolChoice::Auto,
                None,
            )),
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };

        clickhouse
            .insert_datapoints(&[DatapointInsert::Chat(datapoint)])
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        let retrieved_datapoint = clickhouse
            .get_datapoint(&GetDatapointParams {
                dataset_name: dataset_name.clone(),
                datapoint_id,
                allow_stale: None,
            })
            .await
            .unwrap();

        if let StoredDatapoint::Chat(chat_dp) = retrieved_datapoint {
            assert!(chat_dp.tool_params.is_some());
            let tool_params = chat_dp.tool_params.unwrap();

            // Verify provider tools are preserved (previously would have been lost!)
            assert_eq!(tool_params.dynamic_provider_tools.len(), 1);
            if let ProviderToolScope::ModelProvider(mp) =
                &tool_params.dynamic_provider_tools[0].scope
            {
                assert_eq!(mp.model_name, "gpt-4");
                assert_eq!(mp.provider_name, Some("openai".to_string()));
            } else {
                panic!("Expected ModelProvider scope");
            }
            assert_eq!(
                tool_params.dynamic_provider_tools[0].tool["type"],
                "code_interpreter"
            );
        } else {
            panic!("Expected chat datapoint");
        }
    }

    #[tokio::test]
    async fn test_tool_call_storage_function_default_choice() {
        // Test Case 5: AllowedToolsChoice::FunctionDefault
        let clickhouse = get_clickhouse().await;
        let datapoint_id = Uuid::now_v7();
        let dataset_name = format!("test_tool_storage_{}", Uuid::now_v7());

        let datapoint = ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "test_function".to_string(),
            id: datapoint_id,
            name: None,
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "response".to_string(),
            })]),
            tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
                vec![],
                vec![],
                AllowedTools {
                    tools: vec!["func_tool_1".to_string()],
                    choice: AllowedToolsChoice::FunctionDefault, // Key: use function defaults
                },
                ToolChoice::None,
                Some(false),
            )),
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };

        clickhouse
            .insert_datapoints(&[DatapointInsert::Chat(datapoint)])
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        let retrieved_datapoint = clickhouse
            .get_datapoint(&GetDatapointParams {
                dataset_name: dataset_name.clone(),
                datapoint_id,
                allow_stale: None,
            })
            .await
            .unwrap();

        if let StoredDatapoint::Chat(chat_dp) = retrieved_datapoint {
            assert!(chat_dp.tool_params.is_some());
            let tool_params = chat_dp.tool_params.unwrap();

            // FunctionDefault should preserve the choice
            assert_eq!(
                tool_params.allowed_tools.choice,
                AllowedToolsChoice::FunctionDefault
            );
            assert_eq!(tool_params.tool_choice, ToolChoice::None);
            assert_eq!(tool_params.parallel_tool_calls, Some(false));
        } else {
            panic!("Expected chat datapoint");
        }
    }

    #[tokio::test]
    async fn test_tool_call_storage_dynamic_allowed_tools_choice() {
        // Test Case 6: AllowedToolsChoice::AllAllowedTools
        let clickhouse = get_clickhouse().await;
        let datapoint_id = Uuid::now_v7();
        let dataset_name = format!("test_tool_storage_{}", Uuid::now_v7());

        let datapoint = ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "test_function".to_string(),
            id: datapoint_id,
            name: None,
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "response".to_string(),
            })]),
            tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
                vec![],
                vec![],
                AllowedTools {
                    tools: vec!["explicit_tool_1".to_string(), "explicit_tool_2".to_string()],
                    choice: AllowedToolsChoice::Explicit, // Explicit list
                },
                ToolChoice::Specific("explicit_tool_1".to_string()),
                None,
            )),
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };

        clickhouse
            .insert_datapoints(&[DatapointInsert::Chat(datapoint)])
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        let retrieved_datapoint = clickhouse
            .get_datapoint(&GetDatapointParams {
                dataset_name: dataset_name.clone(),
                datapoint_id,
                allow_stale: None,
            })
            .await
            .unwrap();

        if let StoredDatapoint::Chat(chat_dp) = retrieved_datapoint {
            assert!(chat_dp.tool_params.is_some());
            let tool_params = chat_dp.tool_params.unwrap();

            // DynamicAllowedTools should preserve the explicit list
            assert_eq!(
                tool_params.allowed_tools.choice,
                AllowedToolsChoice::Explicit
            );
            assert_eq!(tool_params.allowed_tools.tools.len(), 2);
            assert!(tool_params
                .allowed_tools
                .tools
                .contains(&"explicit_tool_1".to_string()));
            assert!(tool_params
                .allowed_tools
                .tools
                .contains(&"explicit_tool_2".to_string()));

            if let ToolChoice::Specific(tool_name) = tool_params.tool_choice {
                assert_eq!(tool_name, "explicit_tool_1");
            } else {
                panic!("Expected Specific tool choice");
            }
        } else {
            panic!("Expected chat datapoint");
        }
    }

    #[tokio::test]
    async fn test_tool_call_storage_empty_none() {
        // Test Case 7: Empty/None tool params
        let clickhouse = get_clickhouse().await;
        let datapoint_id = Uuid::now_v7();
        let dataset_name = format!("test_tool_storage_{}", Uuid::now_v7());

        let datapoint = ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "test_function".to_string(),
            id: datapoint_id,
            name: None,
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "response".to_string(),
            })]),
            tool_params: None, // No tools at all
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };

        clickhouse
            .insert_datapoints(&[DatapointInsert::Chat(datapoint)])
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        let retrieved_datapoint = clickhouse
            .get_datapoint(&GetDatapointParams {
                dataset_name: dataset_name.clone(),
                datapoint_id,
                allow_stale: None,
            })
            .await
            .unwrap();

        if let StoredDatapoint::Chat(chat_dp) = retrieved_datapoint {
            // Should have no tool params
            assert!(chat_dp.tool_params.is_none());
        } else {
            panic!("Expected chat datapoint");
        }
    }

    #[tokio::test]
    async fn test_tool_call_storage_roundtrip_comprehensive() {
        // Test Case 8: Comprehensive roundtrip test
        let clickhouse = get_clickhouse().await;
        let datapoint_id = Uuid::now_v7();
        let dataset_name = format!("test_tool_storage_{}", Uuid::now_v7());

        let dynamic_tool1 = Tool::Function(FunctionTool {
            name: "dynamic_tool_1".to_string(),
            description: "First dynamic tool".to_string(),
            parameters: json!({"type": "object", "properties": {"param1": {"type": "string"}}}),
            strict: false,
        });

        let dynamic_tool2 = Tool::Function(FunctionTool {
            name: "dynamic_tool_2".to_string(),
            description: "Second dynamic tool".to_string(),
            parameters: json!({"type": "object", "properties": {"param2": {"type": "number"}}}),
            strict: true,
        });

        let provider_tool = ProviderTool {
            scope: ProviderToolScope::ModelProvider(ProviderToolScopeModelProvider {
                model_name: "claude-3-opus".to_string(),
                provider_name: Some("anthropic".to_string()),
            }),
            tool: json!({
                "type": "computer_use"
            }),
        };

        let datapoint = ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "test_function".to_string(),
            id: datapoint_id,
            name: None,
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "response".to_string(),
            })]),
            tool_params: Some(ToolCallConfigDatabaseInsert::new_for_test(
                vec![dynamic_tool1, dynamic_tool2],
                vec![provider_tool],
                AllowedTools {
                    tools: vec!["static_1".to_string(), "static_2".to_string()],
                    choice: AllowedToolsChoice::Explicit,
                },
                ToolChoice::Required,
                Some(true),
            )),
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        };

        clickhouse
            .insert_datapoints(&[DatapointInsert::Chat(datapoint)])
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        let retrieved_datapoint = clickhouse
            .get_datapoint(&GetDatapointParams {
                dataset_name: dataset_name.clone(),
                datapoint_id,
                allow_stale: None,
            })
            .await
            .unwrap();

        if let StoredDatapoint::Chat(chat_dp) = retrieved_datapoint {
            assert!(chat_dp.tool_params.is_some());
            let tool_params = chat_dp.tool_params.unwrap();

            // Verify all components survived the roundtrip
            assert_eq!(tool_params.dynamic_tools.len(), 2);
            assert_eq!(tool_params.dynamic_provider_tools.len(), 1);

            assert_eq!(tool_params.allowed_tools.tools.len(), 2);
            assert!(tool_params
                .allowed_tools
                .tools
                .contains(&"static_1".to_string()));
            assert!(tool_params
                .allowed_tools
                .tools
                .contains(&"static_2".to_string()));

            assert_eq!(tool_params.tool_choice, ToolChoice::Required);
            assert_eq!(tool_params.parallel_tool_calls, Some(true));
        } else {
            panic!("Expected chat datapoint");
        }
    }
}
