//! Tool for uploading a dataset to S3 as Parquet.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use arrow::datatypes::{DataType, Field};
use async_trait::async_trait;
use durable_tools::tensorzero_client::TensorZeroClient;
use durable_tools::{NonControlToolError, TaskTool, ToolContext, ToolMetadata, ToolResult};
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use object_store::buffered::BufWriter as ObjectStoreBufWriter;
use object_store::path::Path as ObjectStorePath;
use parquet::arrow::AsyncArrowWriter;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use serde_arrow::ArrayBuilder;
use tensorzero::ListDatapointsRequest;

use autopilot_client::AutopilotSideInfo;
use durable_tools::tensorzero_client::{S3UploadRequest, S3UploadResponse};

/// Parameters for the upload_dataset tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UploadDatasetToolParams {
    /// The name of the dataset to upload.
    pub dataset_name: String,
    /// Optional maximum number of rows to upload.
    #[serde(default)]
    pub row_limit: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UploadDatasetFormat {
    /// An initial naive format, which just stores the serialized JSON in a single column
    /// This is still an improvement over JSONL, as we get automatic multipart upload and compression
    V1,
}

impl UploadDatasetFormat {
    pub fn schema(&self) -> arrow::datatypes::Schema {
        match self {
            UploadDatasetFormat::V1 => {
                let fields = vec![Field::new("serialized_datapoint", DataType::Utf8, true)];
                arrow::datatypes::Schema::new(fields.clone())
            }
        }
    }
}

/// Output of the upload_dataset tool.
#[derive(Debug, Serialize, Deserialize)]
pub struct UploadDatasetToolOutput {
    /// The S3 URI where the dataset was uploaded (e.g., `s3://bucket/path/to/dataset.parquet`).
    pub s3_uri: String,
    /// Total number of rows uploaded.
    pub total_rows: u32,
    pub format: UploadDatasetFormat,
}

/// Tool for uploading a dataset to S3 as Parquet.
///
/// This tool paginates through a dataset's datapoints, converts each page to
/// an Arrow RecordBatch via `serde_arrow`, and streams them to S3 as a Parquet
/// file using multipart upload. Only one page of data is held in memory at a time.
#[derive(Default)]
pub struct UploadDatasetTool;

/// Page size for listing datapoints during upload.
const PAGE_SIZE: u32 = 1000;

/// Paginate through a dataset's datapoints, convert to Parquet, and upload to
/// the given object store.
///
/// Returns the total number of rows written.
pub async fn upload_dataset_parquet(
    store: Arc<dyn ObjectStore>,
    path: &ObjectStorePath,
    dataset_name: &str,
    row_limit: Option<u32>,
    client: &dyn TensorZeroClient,
) -> Result<u32, anyhow::Error> {
    let schema = Arc::new(UploadDatasetFormat::V1.schema());

    // Initialize the Parquet writer
    let buf_writer = ObjectStoreBufWriter::new(store, path.clone());
    let props = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::ZSTD(
            ZstdLevel::try_new(3)
                .map_err(|e| anyhow::Error::msg(format!("Failed to create zstd level: {e}")))?,
        ))
        .build();
    let mut writer = AsyncArrowWriter::try_new(buf_writer, schema.clone(), Some(props))
        .map_err(|e| anyhow::Error::msg(format!("Failed to create Parquet writer: {e}")))?;

    // Paginate through all pages
    let mut offset = 0u32;
    loop {
        // Calculate page size: always use PAGE_SIZE, but cap to remaining row_limit
        let page_size = match row_limit {
            Some(limit) => {
                let remaining = limit.saturating_sub(offset);
                if remaining == 0 {
                    break;
                }
                PAGE_SIZE.min(remaining)
            }
            None => PAGE_SIZE,
        };

        let response = client
            .list_datapoints(
                dataset_name.to_string(),
                ListDatapointsRequest {
                    limit: Some(page_size),
                    offset: Some(offset),
                    ..Default::default()
                },
            )
            .await
            .map_err(|e| anyhow::Error::msg(format!("Failed to list datapoints: {e}")))?;

        let page_count = response.datapoints.len() as u32;
        if page_count == 0 {
            break;
        }

        let mut builder = ArrayBuilder::from_arrow(schema.fields())?;
        for datapoint in response.datapoints {
            let serialized = serde_json::to_string(&datapoint)
                .map_err(|e| anyhow::Error::msg(format!("Failed to serialize datapoint: {e}")))?;
            builder
                .push(serde_json::json!({
                    "serialized_datapoint": &serialized
                }))
                .map_err(|e| anyhow::Error::msg(format!("Failed to push datapoint: {e}")))?;
        }
        let batch = builder
            .to_record_batch()
            .map_err(|e| anyhow::Error::msg(format!("Failed to convert to RecordBatch: {e}")))?;

        writer
            .write(&batch)
            .await
            .map_err(|e| anyhow::Error::msg(format!("Failed to write Parquet batch: {e}")))?;

        writer
            .flush()
            .await
            .map_err(|e| anyhow::Error::msg(format!("Failed to flush row group: {e}")))?;

        offset += page_count;

        if page_count < page_size {
            break;
        }
    }

    writer
        .close()
        .await
        .map_err(|e| anyhow::Error::msg(format!("Failed to finalize Parquet file: {e}")))?;

    Ok(offset)
}

impl ToolMetadata for UploadDatasetTool {
    type SideInfo = AutopilotSideInfo;
    type Output = UploadDatasetToolOutput;
    type LlmParams = UploadDatasetToolParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("upload_dataset")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Upload a dataset to S3 as Parquet. Paginates through datapoints \
             and uploads the result as a Parquet file.",
        )
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(3600) // 1 hour
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Upload a dataset to S3 as Parquet.",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "The name of the dataset to upload."
                },
                "row_limit": {
                    "type": "integer",
                    "description": "Optional maximum number of rows to upload."
                }
            },
            "required": ["dataset_name"],
            "additionalProperties": false
        });

        serde_json::from_value(schema).map_err(|e| {
            NonControlToolError::SchemaGeneration {
                message: e.to_string(),
            }
            .into()
        })
    }
}

#[async_trait]
impl TaskTool for UploadDatasetTool {
    type ExtraState = ();

    async fn execute(
        &self,
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: &mut ToolContext<'_>,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        // Step 1: Get S3 credentials
        let credentials: S3UploadResponse = ctx
            .step(
                "get_credentials",
                side_info.tool_call_event_id,
                |tool_call_event_id, state| async move {
                    let response = state
                        .t0_client()
                        .s3_initiate_upload(S3UploadRequest { tool_call_event_id })
                        .await
                        .map_err(|e| anyhow::Error::msg(e.to_string()))?;
                    Ok(response)
                },
            )
            .await?;

        // Step 2: Upload dataset to S3
        let dataset_name = llm_params.dataset_name.clone();
        let row_limit = llm_params.row_limit;

        let output: UploadDatasetToolOutput = ctx
            .step(
                "upload",
                (credentials.clone(), dataset_name.clone(), row_limit),
                |params, state| async move {
                    let (creds, dataset_name, row_limit) = params;

                    // Build S3 client with temporary credentials
                    let mut builder = AmazonS3Builder::new()
                        .with_bucket_name(&creds.bucket)
                        .with_region(&creds.region);
                    if let Some(ref key) = creds.access_key_id {
                        builder = builder.with_access_key_id(key);
                    }
                    if let Some(ref secret) = creds.secret_access_key {
                        builder = builder.with_secret_access_key(secret);
                    }
                    if let Some(ref token) = creds.session_token {
                        builder = builder.with_token(token);
                    }
                    let s3: Arc<dyn ObjectStore> = Arc::new(builder.build().map_err(|e| {
                        anyhow::Error::msg(format!("Failed to build S3 client: {e}"))
                    })?);

                    let path = ObjectStorePath::from(creds.key.clone());

                    let total_rows = upload_dataset_parquet(
                        s3,
                        &path,
                        &dataset_name,
                        row_limit,
                        state.t0_client().as_ref(),
                    )
                    .await?;

                    let s3_uri = format!("s3://{}/{}", creds.bucket, creds.key);

                    Ok(UploadDatasetToolOutput {
                        s3_uri,
                        total_rows,
                        format: UploadDatasetFormat::V1,
                    })
                },
            )
            .await?;

        Ok(output)
    }
}
