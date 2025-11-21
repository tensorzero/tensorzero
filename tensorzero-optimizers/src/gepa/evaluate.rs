use tensorzero_core::{
    config::Config,
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::datasets::v1::{
        create_datapoints,
        types::{CreateDatapointRequest, CreateDatapointsRequest},
    },
    error::Error,
    http::TensorzeroHttpClient,
    stored_inference::RenderedSample,
};

/// Create an evaluation dataset from rendered samples
///
/// Uses the datasets v1 API to create datapoints in ClickHouse.
/// Converts each RenderedSample to a CreateDatapointRequest using the
/// `into_create_datapoint_request()` method, which handles type discrimination
/// and validation for both Chat and JSON functions.
///
/// # Arguments
/// * `config` - The TensorZero configuration
/// * `http_client` - The HTTP client for fetching resources
/// * `clickhouse_connection_info` - The ClickHouse connection info
/// * `samples` - The rendered samples to convert into datapoints
/// * `dataset_name` - The name of the dataset to create
///
/// # Returns
/// * `()` - Returns success or error
pub async fn create_evaluation_dataset(
    config: &Config,
    http_client: &TensorzeroHttpClient,
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    samples: Vec<RenderedSample>,
    dataset_name: &str,
) -> Result<(), Error> {
    // Convert RenderedSamples to CreateDatapointRequest using the helper method
    let datapoints: Result<Vec<CreateDatapointRequest>, Error> = samples
        .into_iter()
        .map(|sample| sample.into_create_datapoint_request())
        .collect();

    let request = CreateDatapointsRequest {
        datapoints: datapoints?,
    };

    // Call the datasets v1 create_datapoints function
    create_datapoints(
        config,
        http_client,
        clickhouse_connection_info,
        dataset_name,
        request,
    )
    .await?;

    Ok(())
}
