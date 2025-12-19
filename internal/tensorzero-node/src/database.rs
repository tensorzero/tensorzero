use serde::Deserialize;
use tensorzero::{ClickHouseConnection, setup_clickhouse_without_config};
use uuid::Uuid;

#[napi(js_name = "DatabaseClient")]
pub struct DatabaseClient(Box<dyn ClickHouseConnection>);

#[napi]
impl DatabaseClient {
    #[napi(factory)]
    pub async fn from_clickhouse_url(clickhouse_url: String) -> Result<Self, napi::Error> {
        let connection = setup_clickhouse_without_config(clickhouse_url)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(Self(Box::new(connection)))
    }

    #[napi]
    pub async fn query_demonstration_feedback_by_inference_id(
        &self,
        params: String,
    ) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            query_demonstration_feedback_by_inference_id,
            params,
            QueryDemonstrationFeedbackByInferenceIdParams {
                inference_id,
                before,
                after,
                limit
            }
        )
    }
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
struct QueryDemonstrationFeedbackByInferenceIdParams {
    inference_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: Option<u32>,
}
