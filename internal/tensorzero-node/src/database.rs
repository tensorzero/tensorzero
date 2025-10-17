use serde::Deserialize;
use tensorzero::{
    setup_clickhouse_without_config, ClickHouseConnection, CountDatapointsForDatasetFunctionParams,
    DatapointInsert, DatasetQueryParams, GetAdjacentDatapointIdsParams, GetDatapointParams,
    GetDatasetMetadataParams, GetDatasetRowsParams, StaleDatapointParams, TimeWindow,
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
    pub async fn get_feedback_timeseries(&self, params: String) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            get_feedback_timeseries,
            params,
            GetFeedbackTimeseriesParams {
                function_name,
                metric_name,
                variant_names,
                time_window,
                max_periods
            }
        )
    }

    #[napi]
    pub async fn count_rows_for_dataset(&self, params: String) -> Result<u32, napi::Error> {
        napi_call_no_deserializing!(&self, count_rows_for_dataset, params, DatasetQueryParams)
    }

    #[napi]
    pub async fn insert_rows_for_dataset(&self, params: String) -> Result<u32, napi::Error> {
        napi_call_no_deserializing!(&self, insert_rows_for_dataset, params, DatasetQueryParams)
    }

    #[napi]
    pub async fn get_dataset_rows(&self, params: String) -> Result<String, napi::Error> {
        napi_call!(&self, get_dataset_rows, params, GetDatasetRowsParams)
    }

    #[napi]
    pub async fn get_dataset_metadata(&self, params: String) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            get_dataset_metadata,
            params,
            GetDatasetMetadataParams
        )
    }

    #[napi]
    pub async fn count_datasets(&self) -> Result<u32, napi::Error> {
        napi_call_no_deserializing!(&self, count_datasets)
    }

    #[napi]
    pub async fn stale_datapoint(&self, params: String) -> Result<(), napi::Error> {
        napi_call_no_deserializing!(&self, stale_datapoint, params, StaleDatapointParams)
    }

    #[napi]
    pub async fn insert_datapoint(&self, params: String) -> Result<(), napi::Error> {
        napi_call_no_deserializing!(&self, insert_datapoint, params, DatapointInsert)
    }

    #[napi]
    pub async fn count_datapoints_for_dataset_function(
        &self,
        params: String,
    ) -> Result<u32, napi::Error> {
        napi_call_no_deserializing!(
            &self,
            count_datapoints_for_dataset_function,
            params,
            CountDatapointsForDatasetFunctionParams
        )
    }

    #[napi]
    pub async fn get_adjacent_datapoint_ids(&self, params: String) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            get_adjacent_datapoint_ids,
            params,
            GetAdjacentDatapointIdsParams
        )
    }

    #[napi]
    pub async fn get_datapoint(&self, params: String) -> Result<String, napi::Error> {
        napi_call!(&self, get_datapoint, params, GetDatapointParams)
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
    pub time_window: TimeWindow,
    pub max_periods: u32,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
struct GetFeedbackByVariantParams {
    pub metric_name: String,
    pub function_name: String,
    pub variant_names: Option<Vec<String>>,
}
