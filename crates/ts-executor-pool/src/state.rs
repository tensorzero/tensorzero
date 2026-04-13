/// Configuration for uploading V8 heap snapshots on deno OOM.
#[derive(Clone, Debug)]
pub struct OomSnapshotConfig {
    pub bucket_name: String,
    pub bucket_region: String,
    pub s3_client: aws_sdk_s3::Client,
}
