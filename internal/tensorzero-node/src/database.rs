use serde::Deserialize;
use tensorzero::{
    setup_clickhouse_without_config, ClickHouseConnection, FunctionType,
    QueryInferenceTableBoundsParams, QueryInferenceTableParams, TimeWindow,
};
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
    pub async fn query_inference_table(&self, params: String) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            query_inference_table,
            params,
            QueryInferenceTableParams
        )
    }

    #[napi]
    pub async fn query_inference_table_bounds(
        &self,
        params: String,
    ) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            query_inference_table_bounds,
            params,
            QueryInferenceTableBoundsParams
        )
    }

    #[napi]
    pub async fn count_inferences_for_function(&self, params: String) -> Result<u32, napi::Error> {
        napi_call_no_deserializing!(
            &self,
            count_inferences_for_function,
            params,
            CountInferencesForFunctionParams {
                function_name,
                function_type
            }
        )
    }

    #[napi]
    pub async fn count_inferences_for_variant(&self, params: String) -> Result<u32, napi::Error> {
        napi_call_no_deserializing!(
            &self,
            count_inferences_for_variant,
            params,
            CountInferencesForVariantParams {
                function_name,
                function_type,
                variant_name
            }
        )
    }

    #[napi]
    pub async fn count_inferences_for_episode(&self, params: String) -> Result<u32, napi::Error> {
        napi_call_no_deserializing!(
            &self,
            count_inferences_for_episode,
            params,
            CountInferencesForEpisodeParams { episode_id }
        )
    }

    #[napi]
    pub async fn query_inference_by_id(&self, params: String) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            query_inference_by_id,
            params,
            QueryInferenceByIdParams { id }
        )
    }

    #[napi]
    pub async fn query_model_inferences_by_inference_id(
        &self,
        params: String,
    ) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            query_model_inferences_by_inference_id,
            params,
            QueryModelInferencesParams { id }
        )
    }

    #[napi]
    pub async fn count_inferences_by_function(&self) -> Result<String, napi::Error> {
        napi_call!(&self, count_inferences_by_function)
    }

    #[napi]
    pub async fn get_adjacent_inference_ids(&self, params: String) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            get_adjacent_inference_ids,
            params,
            GetAdjacentInferenceParams {
                current_inference_id
            }
        )
    }

    #[napi]
    pub async fn get_adjacent_episode_ids(&self, params: String) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            get_adjacent_episode_ids,
            params,
            GetAdjacentEpisodeParams { current_episode_id }
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
#[ts(export)]
struct CountInferencesForFunctionParams {
    pub function_name: String,
    pub function_type: FunctionType,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export)]
struct CountInferencesForVariantParams {
    pub function_name: String,
    pub function_type: FunctionType,
    pub variant_name: String,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export)]
struct CountInferencesForEpisodeParams {
    pub episode_id: Uuid,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export)]
struct QueryInferenceByIdParams {
    pub id: Uuid,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export)]
struct QueryModelInferencesParams {
    pub id: Uuid,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export)]
struct GetAdjacentInferenceParams {
    pub current_inference_id: Uuid,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export)]
struct GetAdjacentEpisodeParams {
    pub current_episode_id: Uuid,
}
