mod common;

use std::sync::Arc;

use arrow::array::StringArray;
use arrow::datatypes::FieldRef;
use object_store::memory::InMemory;
use object_store::path::Path as ObjectStorePath;
use object_store::{ObjectStore, ObjectStoreExt};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use uuid::Uuid;

use autopilot_tools::tools::upload_dataset_parquet;
use common::{
    MockTensorZeroClient, create_mock_chat_datapoint, create_mock_get_datapoints_response,
};

#[tokio::test]
async fn test_upload_dataset_parquet_schema_and_first_row() {
    let datapoint_id = Uuid::now_v7();
    let datapoint = create_mock_chat_datapoint(datapoint_id, "test_dataset", "test_function");
    let mock_response = create_mock_get_datapoints_response(vec![datapoint.clone()]);

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_list_datapoints()
        .withf(|name, _| name == "test_dataset")
        .return_once(move |_, _| Ok(mock_response));

    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = ObjectStorePath::from("test/dataset.parquet");

    let total_rows = upload_dataset_parquet(
        Arc::clone(&store),
        &path,
        "test_dataset",
        None,
        &mock_client,
    )
    .await
    .expect("upload_dataset_parquet should succeed");

    assert_eq!(total_rows, 1, "Expected 1 row to be uploaded");

    // Read back the Parquet file from the in-memory store
    let get_result = store
        .get(&path)
        .await
        .expect("Parquet file should exist in the store");
    let data = get_result
        .bytes()
        .await
        .expect("Should be able to read Parquet bytes");

    let reader = ParquetRecordBatchReaderBuilder::try_new(data)
        .expect("Should be able to create Parquet reader")
        .build()
        .expect("Should be able to build Parquet reader");

    let batches: Vec<_> = reader
        .collect::<Result<Vec<_>, _>>()
        .expect("Should be able to read all record batches");
    assert_eq!(batches.len(), 1, "Expected exactly 1 row group");

    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 1, "Expected 1 row in the batch");

    // Verify exact schema field names
    let schema = batch.schema();
    let field_names: Vec<&str> = schema
        .fields()
        .iter()
        .map(|f: &FieldRef| f.name().as_str())
        .collect();
    assert_eq!(
        field_names,
        vec!["serialized_datapoint",],
        "Schema field names should match exactly"
    );

    let serialized_datapoint_col = batch
        .column_by_name("serialized_datapoint")
        .expect("Schema should have `serialized_datapoint` column");

    let serialized_datapoint_arr = serialized_datapoint_col
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("`serialized_datapoint` column should be a LargeStringArray");

    let value = serialized_datapoint_arr.value(0);
    let expected_value =
        serde_json::to_string(&datapoint).expect("Should be able to serialize datapoint");
    assert_eq!(
        value, expected_value,
        "`serialized_datapoint` should match the input datapoint"
    );
}

/// Test uploading >1000 datapoints, which requires multiple pages of pagination
/// (PAGE_SIZE is 1000).
#[tokio::test]
async fn test_upload_dataset_parquet_multi_page() {
    let datapoints_page1: Vec<_> = (0..1000)
        .map(|_| create_mock_chat_datapoint(Uuid::now_v7(), "test_dataset", "test_function"))
        .collect();
    let datapoints_page2: Vec<_> = (0..500)
        .map(|_| create_mock_chat_datapoint(Uuid::now_v7(), "test_dataset", "test_function"))
        .collect();

    // Keep clones to verify serialized content later
    let all_datapoints: Vec<_> = datapoints_page1
        .iter()
        .chain(datapoints_page2.iter())
        .cloned()
        .collect();

    let response1 = create_mock_get_datapoints_response(datapoints_page1);
    let response2 = create_mock_get_datapoints_response(datapoints_page2);

    let mut mock_client = MockTensorZeroClient::new();
    let mut seq = mockall::Sequence::new();

    mock_client
        .expect_list_datapoints()
        .times(1)
        .in_sequence(&mut seq)
        .withf(|name, req| {
            name == "test_dataset" && req.offset == Some(0) && req.limit == Some(1000)
        })
        .return_once(move |_, _| Ok(response1));

    mock_client
        .expect_list_datapoints()
        .times(1)
        .in_sequence(&mut seq)
        .withf(|name, req| {
            name == "test_dataset" && req.offset == Some(1000) && req.limit == Some(1000)
        })
        .return_once(move |_, _| Ok(response2));

    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = ObjectStorePath::from("test/dataset.parquet");

    let total_rows = upload_dataset_parquet(
        Arc::clone(&store),
        &path,
        "test_dataset",
        None,
        &mock_client,
    )
    .await
    .expect("upload_dataset_parquet should succeed");

    assert_eq!(total_rows, 1500, "Expected 1500 rows to be uploaded");

    // Read back the Parquet file and verify row contents
    let get_result = store
        .get(&path)
        .await
        .expect("Parquet file should exist in the store");
    let data = get_result
        .bytes()
        .await
        .expect("Should be able to read Parquet bytes");

    let reader = ParquetRecordBatchReaderBuilder::try_new(data)
        .expect("Should be able to create Parquet reader")
        .build()
        .expect("Should be able to build Parquet reader");

    let batches: Vec<_> = reader
        .collect::<Result<Vec<_>, _>>()
        .expect("Should be able to read all record batches");

    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 1500, "Expected 1500 total rows in Parquet file");

    // Verify each row matches the expected serialized datapoint
    let mut row_idx = 0;
    for batch in &batches {
        let col = batch
            .column_by_name("serialized_datapoint")
            .expect("Should have `serialized_datapoint` column");
        let arr = col
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("`serialized_datapoint` column should be a StringArray");
        for i in 0..batch.num_rows() {
            let expected = serde_json::to_string(&all_datapoints[row_idx])
                .expect("Should be able to serialize datapoint");
            assert_eq!(
                arr.value(i),
                expected,
                "Row {row_idx} `serialized_datapoint` should match the input datapoint"
            );
            row_idx += 1;
        }
    }
}

/// Test that `row_limit` controls the page size used for pagination.
#[tokio::test]
async fn test_upload_dataset_parquet_with_row_limit() {
    let datapoints_page1: Vec<_> = (0..100)
        .map(|_| create_mock_chat_datapoint(Uuid::now_v7(), "test_dataset", "test_function"))
        .collect();
    let datapoints_page2: Vec<_> = (0..100)
        .map(|_| create_mock_chat_datapoint(Uuid::now_v7(), "test_dataset", "test_function"))
        .collect();
    let datapoints_page3: Vec<_> = (0..50)
        .map(|_| create_mock_chat_datapoint(Uuid::now_v7(), "test_dataset", "test_function"))
        .collect();

    // Keep clones to verify serialized content later
    let all_datapoints: Vec<_> = datapoints_page1
        .iter()
        .chain(datapoints_page2.iter())
        .chain(datapoints_page3.iter())
        .cloned()
        .collect();

    let response1 = create_mock_get_datapoints_response(datapoints_page1);
    let response2 = create_mock_get_datapoints_response(datapoints_page2);
    let response3 = create_mock_get_datapoints_response(datapoints_page3);

    let mut mock_client = MockTensorZeroClient::new();
    let mut seq = mockall::Sequence::new();

    // With row_limit=100, the function should request pages of 100
    mock_client
        .expect_list_datapoints()
        .times(1)
        .in_sequence(&mut seq)
        .withf(|name, req| {
            name == "test_dataset" && req.offset == Some(0) && req.limit == Some(100)
        })
        .return_once(move |_, _| Ok(response1));

    mock_client
        .expect_list_datapoints()
        .times(1)
        .in_sequence(&mut seq)
        .withf(|name, req| {
            name == "test_dataset" && req.offset == Some(100) && req.limit == Some(100)
        })
        .return_once(move |_, _| Ok(response2));

    mock_client
        .expect_list_datapoints()
        .times(1)
        .in_sequence(&mut seq)
        .withf(|name, req| {
            name == "test_dataset" && req.offset == Some(200) && req.limit == Some(100)
        })
        .return_once(move |_, _| Ok(response3));

    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = ObjectStorePath::from("test/dataset.parquet");

    let total_rows = upload_dataset_parquet(
        Arc::clone(&store),
        &path,
        "test_dataset",
        Some(100),
        &mock_client,
    )
    .await
    .expect("upload_dataset_parquet should succeed");

    assert_eq!(total_rows, 250, "Expected 250 rows to be uploaded");

    // Read back the Parquet file and verify row contents
    let get_result = store
        .get(&path)
        .await
        .expect("Parquet file should exist in the store");
    let data = get_result
        .bytes()
        .await
        .expect("Should be able to read Parquet bytes");

    let reader = ParquetRecordBatchReaderBuilder::try_new(data)
        .expect("Should be able to create Parquet reader")
        .build()
        .expect("Should be able to build Parquet reader");

    let batches: Vec<_> = reader
        .collect::<Result<Vec<_>, _>>()
        .expect("Should be able to read all record batches");

    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 250, "Expected 250 total rows in Parquet file");

    // Verify each row matches the expected serialized datapoint
    let mut row_idx = 0;
    for batch in &batches {
        let col = batch
            .column_by_name("serialized_datapoint")
            .expect("Should have `serialized_datapoint` column");
        let arr = col
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("`serialized_datapoint` column should be a StringArray");
        for i in 0..batch.num_rows() {
            let expected = serde_json::to_string(&all_datapoints[row_idx])
                .expect("Should be able to serialize datapoint");
            assert_eq!(
                arr.value(i),
                expected,
                "Row {row_idx} `serialized_datapoint` should match the input datapoint"
            );
            row_idx += 1;
        }
    }
}
