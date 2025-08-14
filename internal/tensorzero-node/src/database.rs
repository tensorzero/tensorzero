use tensorzero::{
    setup_clickhouse_without_config, DatabaseConnection, ModelUsageTimePoint, TimeWindow,
};

#[napi(js_name = "DatabaseClient")]
pub struct DatabaseClient(Box<dyn DatabaseConnection>);

#[napi]
impl DatabaseClient {
    #[napi(factory)]
    pub async fn from_clickhouse_url(clickhouse_url: String) -> Result<Self, napi::Error> {
        let connection = setup_clickhouse_without_config(clickhouse_url)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(Self(Box::new(connection)))
    }

    pub async fn get_model_usage_timeseries(
        &self,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> Result<Vec<ModelUsageTimePoint>, napi::Error> {
        let result = self
            .0
            .get_model_usage_timeseries(time_window, max_periods)
            .await;
        result.map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}
