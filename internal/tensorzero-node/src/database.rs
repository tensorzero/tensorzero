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
        napi_call!(
            &self,
            get_model_usage_timeseries,
            params,
            GetModelUsageTimeseriesParams {
                time_window,
                max_periods
            }
        )
    }

    #[napi]
    pub async fn get_model_latency_quantiles(&self, params: String) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            get_model_latency_quantiles,
            params,
            GetModelLatencyQuantilesParams { time_window }
        )
    }

    #[napi]
    pub async fn count_distinct_models_used(&self) -> Result<u32, napi::Error> {
        napi_call_no_deserializing!(&self, count_distinct_models_used)
    }

    #[napi]
    pub async fn query_episode_table(&self, params: String) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            query_episode_table,
            params,
            QueryEpisodeTableParams {
                page_size,
                before,
                after
            }
        )
    }

    #[napi]
    pub async fn query_episode_table_bounds(&self) -> Result<String, napi::Error> {
        napi_call!(&self, query_episode_table_bounds)
    }

    #[napi]
    pub async fn get_feedback_timeseries(&self, params: String) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            get_feedback_timeseries,
            params,
            GetFeedbackTimeseriesParams {
                function_name,
                metric_name,
                variant_names,
                interval_minutes,
                max_periods
            }
        )
    }
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
struct GetModelUsageTimeseriesParams {
    pub time_window: TimeWindow,
    pub max_periods: u32,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
struct GetModelLatencyQuantilesParams {
    pub time_window: TimeWindow,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
struct QueryEpisodeTableParams {
    pub page_size: u32,
    pub before: Option<Uuid>,
    pub after: Option<Uuid>,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
struct GetFeedbackTimeseriesParams {
    pub function_name: String,
    pub metric_name: String,
    pub variant_names: Option<Vec<String>>,
    pub interval_minutes: u32,
    pub max_periods: u32,
}
