use serde_json::json;
use std::collections::HashMap;
use uuid::Uuid;

use tensorzero::{
    Datapoint, DatasetQueryParams, FloatComparisonOperator, GetDatapointParams,
    GetDatasetMetadataParams, Role,
};
use tensorzero_core::config::{MetricConfigLevel, MetricConfigType};
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::datasets::{
    ChatInferenceDatapointInsert, CountDatapointsForDatasetFunctionParams, DatapointInsert,
    DatasetMetadata, DatasetOutputSource, DatasetQueries, GetAdjacentDatapointIdsParams,
    GetDatasetRowsParams, JsonInferenceDatapointInsert, MetricFilter, StaleDatapointParams,
};
use tensorzero_core::endpoints::datasets::DatapointKind;
use tensorzero_core::inference::types::{
    ContentBlockChatOutput, JsonInferenceOutput, StoredInput, StoredInputMessage,
    StoredInputMessageContent, Text,
};
use tensorzero_core::stored_inference::StoredSample;

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
        page_size: None,
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
        page_size: None,
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

#[tokio::test]
async fn test_get_dataset_rows_returns_correct_rows_for_specific_dataset() {
    let params = GetDatasetRowsParams {
        dataset_name: "notadataset".to_string(),
        page_size: 10,
        offset: 0,
    };

    let rows = get_clickhouse()
        .await
        .get_dataset_rows(&params)
        .await
        .unwrap();

    assert!(rows.is_empty(), "Should have 0 rows");
}

#[tokio::test]
async fn test_get_dataset_rows_pages_correctly() {
    let mut all_rows = Vec::new();
    let mut offset = 0;
    let page_size = 10;

    loop {
        let params = GetDatasetRowsParams {
            dataset_name: "foo".to_string(),
            page_size,
            offset,
        };
        let rows = get_clickhouse()
            .await
            .get_dataset_rows(&params)
            .await
            .unwrap();
        let is_last_page = rows.len() != page_size as usize;

        all_rows.extend(rows);
        offset += page_size;

        if is_last_page {
            break;
        }
    }

    // TODO(#3903): Stop making assumptions about what data exists in the database, and
    // make data dependencies explicit in e2e tests, so tests can execute independently
    // and without requiring loading database fixtures.
    assert!(!all_rows.is_empty(), "Should have existing rows");
}

#[tokio::test]
async fn test_count_datasets() {
    let clickhouse = get_clickhouse().await;

    // Get initial count
    let initial_count = clickhouse.count_datasets().await.unwrap();

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
        .insert_datapoint(&DatapointInsert::Chat(datapoint1))
        .await
        .unwrap();
    clickhouse
        .insert_datapoint(&DatapointInsert::Chat(datapoint2))
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Count should increase by 2
    let new_count = clickhouse.count_datasets().await.unwrap();
    assert_eq!(new_count, initial_count + 2);
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
            .insert_datapoint(&DatapointInsert::Chat(datapoint))
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
            .insert_datapoint(&DatapointInsert::Json(datapoint))
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
        .insert_datapoint(&datapoint_insert)
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
        .insert_datapoint(&datapoint_insert)
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
        .insert_datapoint(&DatapointInsert::Chat(datapoint))
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
        .insert_datapoint(&DatapointInsert::Chat(datapoint))
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

    if let Datapoint::Json(datapoint) = datapoint {
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
            content: vec![StoredInputMessageContent::Text {
                value: "Is it a living thing?".to_string().into(),
            }],
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

    if let Datapoint::Chat(datapoint) = datapoint {
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
    clickhouse.insert_datapoint(&chat_datapoint).await.unwrap();

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

    if let Datapoint::Chat(chat_dp) = retrieved_datapoint {
        assert_eq!(chat_dp.source_inference_id, Some(source_inference_id));
    } else {
        panic!("Expected chat datapoint");
    }

    // Test staling
    clickhouse
        .stale_datapoint(&StaleDatapointParams {
            dataset_name: "test_chat_dataset".to_string(),
            datapoint_id,
            function_type: DatapointKind::Chat,
        })
        .await
        .unwrap();

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

    if let Datapoint::Chat(chat_dp) = staled_datapoint {
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
    clickhouse.insert_datapoint(&json_datapoint).await.unwrap();

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

    if let Datapoint::Json(json_dp) = retrieved_datapoint {
        assert_eq!(json_dp.source_inference_id, Some(source_inference_id));
    } else {
        panic!("Expected json datapoint");
    }

    // Test staling
    clickhouse
        .stale_datapoint(&StaleDatapointParams {
            dataset_name: "test_json_dataset".to_string(),
            datapoint_id,
            function_type: DatapointKind::Json,
        })
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

    if let Datapoint::Json(json_dp) = staled_datapoint {
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

    // First insertion
    clickhouse.insert_datapoint(&chat_datapoint).await.unwrap();

    // Second insertion with same ID should not throw
    clickhouse.insert_datapoint(&chat_datapoint).await.unwrap();
}

#[tokio::test]
async fn test_handles_staling_of_non_existent_datapoint() {
    let clickhouse = get_clickhouse().await;

    // Should not throw when trying to stale a non-existent datapoint
    let result = clickhouse
        .stale_datapoint(&StaleDatapointParams {
            dataset_name: "fake".to_string(),
            datapoint_id: Uuid::now_v7(),
            function_type: DatapointKind::Chat,
        })
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

    let result = clickhouse.insert_datapoint(&chat_datapoint).await;
    assert!(
        result.is_err(),
        "Should reject reserved dataset name 'builder'"
    );
}

#[tokio::test]
async fn test_get_adjacent_datapoint_ids() {
    let clickhouse = get_clickhouse().await;

    let dataset_name = format!("test_adjacent_{}", Uuid::now_v7());

    // Insert three datapoints
    let id1 = Uuid::now_v7();
    let id2 = Uuid::now_v7();
    let id3 = Uuid::now_v7();

    for id in [id1, id2, id3] {
        let datapoint = ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "test_function".to_string(),
            id,
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
            .insert_datapoint(&DatapointInsert::Chat(datapoint))
            .await
            .unwrap();
    }

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Test middle datapoint
    let adjacent = clickhouse
        .get_adjacent_datapoint_ids(&GetAdjacentDatapointIdsParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: id2,
        })
        .await
        .unwrap();
    assert_eq!(
        adjacent.previous_id,
        Some(id1),
        "Should return previous id for the middle datapoint"
    );
    assert_eq!(
        adjacent.next_id,
        Some(id3),
        "Should return next id for the middle datapoint"
    );

    // Test first datapoint
    let adjacent = clickhouse
        .get_adjacent_datapoint_ids(&GetAdjacentDatapointIdsParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: id1,
        })
        .await
        .unwrap();
    assert_eq!(
        adjacent.previous_id, None,
        "Should return None as previous id for the first datapoint"
    );
    assert_eq!(
        adjacent.next_id,
        Some(id2),
        "Should return next id for the first datapoint"
    );

    // Test last datapoint
    let adjacent = clickhouse
        .get_adjacent_datapoint_ids(&GetAdjacentDatapointIdsParams {
            dataset_name: dataset_name.clone(),
            datapoint_id: id3,
        })
        .await
        .unwrap();
    assert_eq!(
        adjacent.previous_id,
        Some(id2),
        "Should return previous id for the last datapoint"
    );
    assert_eq!(
        adjacent.next_id, None,
        "Should return None as next id for the last datapoint"
    );
}

#[tokio::test]
async fn test_get_datapoints_with_empty_ids() {
    let clickhouse = get_clickhouse().await;

    let result = clickhouse
        .get_datapoints("test_dataset", &[], false)
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
        .insert_datapoint(&DatapointInsert::Chat(datapoint))
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve using get_datapoints
    let result = clickhouse
        .get_datapoints(&dataset_name, &[datapoint_id], false)
        .await
        .unwrap();

    assert_eq!(result.len(), 1, "Should return exactly one datapoint");

    if let Datapoint::Chat(dp) = &result[0] {
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
        .insert_datapoint(&DatapointInsert::Json(datapoint))
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve using get_datapoints
    let result = clickhouse
        .get_datapoints(&dataset_name, &[datapoint_id], false)
        .await
        .unwrap();

    assert_eq!(result.len(), 1, "Should return exactly one datapoint");

    if let Datapoint::Json(dp) = &result[0] {
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
        .insert_datapoint(&DatapointInsert::Chat(chat_dp1))
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
        .insert_datapoint(&DatapointInsert::Json(json_dp))
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
        .insert_datapoint(&DatapointInsert::Chat(chat_dp2))
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Retrieve all three datapoints
    let result = clickhouse
        .get_datapoints(&dataset_name, &[chat_id1, json_id, chat_id2], false)
        .await
        .unwrap();

    assert_eq!(
        result.len(),
        3,
        "Should return all three datapoints (2 chat, 1 json)"
    );

    // Verify we got all the expected IDs
    let returned_ids: Vec<Uuid> = result.iter().map(Datapoint::id).collect();
    assert!(returned_ids.contains(&chat_id1));
    assert!(returned_ids.contains(&json_id));
    assert!(returned_ids.contains(&chat_id2));

    // Count types
    let chat_count = result
        .iter()
        .filter(|dp| matches!(dp, Datapoint::Chat(_)))
        .count();
    let json_count = result
        .iter()
        .filter(|dp| matches!(dp, Datapoint::Json(_)))
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
        .insert_datapoint(&DatapointInsert::Chat(datapoint))
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Query with both existing and non-existent IDs
    let non_existent_id = Uuid::now_v7();
    let another_non_existent_id = Uuid::now_v7();
    let result = clickhouse
        .get_datapoints(
            &dataset_name,
            &[datapoint_id, non_existent_id, another_non_existent_id],
            false,
        )
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
        .insert_datapoint(&DatapointInsert::Chat(datapoint))
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Verify we can retrieve it before staling
    let result = clickhouse
        .get_datapoints(&dataset_name, &[datapoint_id], false)
        .await
        .unwrap();
    assert_eq!(result.len(), 1, "Should retrieve datapoint before staling");

    // Stale the datapoint
    clickhouse
        .stale_datapoint(&StaleDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id,
            function_type: DatapointKind::Chat,
        })
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Try to retrieve with allow_stale=false
    let result = clickhouse
        .get_datapoints(&dataset_name, &[datapoint_id], false)
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
        .insert_datapoint(&DatapointInsert::Chat(datapoint))
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Stale the datapoint
    clickhouse
        .stale_datapoint(&StaleDatapointParams {
            dataset_name: dataset_name.clone(),
            datapoint_id,
            function_type: DatapointKind::Chat,
        })
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Try to retrieve with allow_stale=true
    let result = clickhouse
        .get_datapoints(&dataset_name, &[datapoint_id], true)
        .await
        .unwrap();

    assert_eq!(
        result.len(),
        1,
        "Should return staled datapoint when allow_stale=true"
    );

    if let Datapoint::Chat(dp) = &result[0] {
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
        .insert_datapoint(&DatapointInsert::Chat(datapoint))
        .await
        .unwrap();

    // Sleep for 1 second for ClickHouse to become consistent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Try to retrieve with different dataset name
    let wrong_dataset = format!("wrong_{dataset_name}");
    let result = clickhouse
        .get_datapoints(&wrong_dataset, &[datapoint_id], false)
        .await
        .unwrap();

    assert_eq!(
        result.len(),
        0,
        "Should not return datapoint when querying wrong dataset"
    );
}
