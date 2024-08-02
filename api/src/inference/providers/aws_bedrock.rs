use tokio::sync::OnceCell;

use crate::error::Error;
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{
    InferenceResponseStream, ModelInferenceRequest, ModelInferenceResponse,
    ModelInferenceResponseChunk,
};
use crate::model::ProviderConfig;

static AWS_BEDROCK_CLIENT: OnceCell<aws_sdk_bedrockruntime::Client> = OnceCell::const_new();

async fn get_aws_bedrock_client() -> &'static aws_sdk_bedrockruntime::Client {
    AWS_BEDROCK_CLIENT
        .get_or_init(|| async {
            let config = aws_config::load_from_env().await;
            aws_sdk_bedrockruntime::Client::new(&config)
        })
        .await
}

pub struct AwsBedrockProvider;

impl InferenceProvider for AwsBedrockProvider {
    async fn infer<'a>(
        request: &'a ModelInferenceRequest<'a>,
        config: &'a ProviderConfig,
        http_client: &'a reqwest::Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let aws_bedrock_client = get_aws_bedrock_client().await;
        todo!()
    }

    async fn infer_stream<'a>(
        request: &'a ModelInferenceRequest<'a>,
        config: &'a ProviderConfig,
        http_client: &'a reqwest::Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        todo!()
    }
}
