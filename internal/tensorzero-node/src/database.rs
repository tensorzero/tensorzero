use serde::Deserialize;
use tensorzero::{setup_clickhouse_without_config, ClickHouseConnection, TimeWindow};
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
    pub async fn get_model_usage_timeseries(&self, params: String) -> Result<String, napi::Error> {
        // TODO: (Aaron?, Viraj?) should we serialize here?
        // I think we could use native types if we cfged #[napi] into the core codebase.
        // It will be potentially problematic to serialize every database call twice
        let GetModelUsageTimeseriesParams {
            time_window,
            max_periods,
        } = serde_json::from_str(&params).map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let result = self
            .0
            .get_model_usage_timeseries(time_window, max_periods)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_string(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub async fn get_model_latency_quantiles(&self, params: String) -> Result<String, napi::Error> {
        let GetModelLatencyQuantilesParams { time_window } =
            serde_json::from_str(&params).map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let result = self
            .0
            .get_model_latency_quantiles(time_window)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_string(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub async fn count_distinct_models_used(&self) -> Result<u32, napi::Error> {
        let result = self
            .0
            .count_distinct_models_used()
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(result)
    }

    #[napi]
    pub async fn query_episode_table(&self, params: String) -> Result<String, napi::Error> {
        let QueryEpisodeTableParams {
            page_size,
            before,
            after,
        } = serde_json::from_str(&params).map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let result = self
            .0
            .query_episode_table(page_size, before, after)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_string(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub async fn query_episode_table_bounds(&self) -> Result<String, napi::Error> {
        let bounds = self
            .0
            .query_episode_table_bounds()
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_string(&bounds).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub async fn query_feedback_by_target_id(&self, params: String) -> Result<String, napi::Error> {
        let QueryFeedbackByTargetIdParams {
            target_id,
            before,
            after,
            page_size,
        } = serde_json::from_str(&params).map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let result = self
            .0
            .query_feedback_by_target_id(target_id, before, after, page_size)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_string(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub async fn query_feedback_bounds_by_target_id(
        &self,
        params: String,
    ) -> Result<String, napi::Error> {
        let QueryFeedbackBoundsByTargetIdParams { target_id } =
            serde_json::from_str(&params).map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let result = self
            .0
            .query_feedback_bounds_by_target_id(target_id)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_string(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub async fn count_feedback_by_target_id(&self, params: String) -> Result<String, napi::Error> {
        let CountFeedbackByTargetIdParams { target_id } =
            serde_json::from_str(&params).map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let result = self
            .0
            .count_feedback_by_target_id(target_id)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_string(&result).map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

#[derive(Deserialize)]
struct GetModelUsageTimeseriesParams {
    time_window: TimeWindow,
    max_periods: u32,
}

#[derive(Deserialize)]
struct GetModelLatencyQuantilesParams {
    time_window: TimeWindow,
}

#[derive(Deserialize)]
struct QueryEpisodeTableParams {
    page_size: u32,
    before: Option<Uuid>,
    after: Option<Uuid>,
}

#[derive(Deserialize)]
struct QueryFeedbackByTargetIdParams {
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    page_size: Option<u32>,
}

#[derive(Deserialize)]
struct QueryFeedbackBoundsByTargetIdParams {
    target_id: Uuid,
}

#[derive(Deserialize)]
struct CountFeedbackByTargetIdParams {
    target_id: Uuid,
}
