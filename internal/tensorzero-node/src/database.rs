use serde::Deserialize;
use tensorzero::{ClickHouseConnection, TimeWindow, setup_clickhouse_without_config};
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
    pub async fn get_cumulative_feedback_timeseries(
        &self,
        params: String,
    ) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            get_cumulative_feedback_timeseries,
            params,
            GetCumulativeFeedbackTimeseriesParams {
                function_name,
                metric_name,
                variant_names,
                time_window,
                max_periods
            }
        )
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

    #[napi]
    pub async fn get_feedback_by_variant(&self, params: String) -> Result<String, napi::Error> {
        let params_struct: GetFeedbackByVariantParams =
            serde_json::from_str(&params).map_err(|e| napi::Error::from_reason(e.to_string()))?;

        let result = self
            .0
            .get_feedback_by_variant(
                &params_struct.metric_name,
                &params_struct.function_name,
                params_struct.variant_names.as_ref(),
            )
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;

        serde_json::to_string(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
struct GetCumulativeFeedbackTimeseriesParams {
    pub function_name: String,
    pub metric_name: String,
    pub variant_names: Option<Vec<String>>,
    pub time_window: TimeWindow,
    pub max_periods: u32,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
struct QueryDemonstrationFeedbackByInferenceIdParams {
    inference_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: Option<u32>,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
struct GetFeedbackByVariantParams {
    metric_name: String,
    function_name: String,
    #[ts(optional)]
    variant_names: Option<Vec<String>>,
}
