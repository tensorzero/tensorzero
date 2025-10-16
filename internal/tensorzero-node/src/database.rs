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
    pub async fn query_feedback_by_target_id(&self, params: String) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            query_feedback_by_target_id,
            params,
            QueryFeedbackByTargetIdParams {
                target_id,
                before,
                after,
                page_size
            }
        )
    }

    #[napi]
    pub async fn query_feedback_bounds_by_target_id(
        &self,
        params: String,
    ) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            query_feedback_bounds_by_target_id,
            params,
            QueryFeedbackBoundsByTargetIdParams { target_id }
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
    pub async fn count_feedback_by_target_id(&self, params: String) -> Result<String, napi::Error> {
        napi_call!(
            &self,
            count_feedback_by_target_id,
            params,
            CountFeedbackByTargetIdParams { target_id }
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
                page_size
            }
        )
    }

    #[napi]
    pub async fn get_datapoint(&self, params: String) -> Result<String, napi::Error> {
        napi_call!(&self, get_datapoint, params, GetDatapointParams)
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
    #[ts(optional)]
    pub before: Option<Uuid>,
    #[ts(optional)]
    pub after: Option<Uuid>,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
struct QueryFeedbackByTargetIdParams {
    target_id: Uuid,
    #[ts(optional)]
    before: Option<Uuid>,
    #[ts(optional)]
    after: Option<Uuid>,
    page_size: Option<u32>,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export)]
struct QueryDemonstrationFeedbackByInferenceIdParams {
    inference_id: Uuid,
    #[ts(optional)]
    before: Option<Uuid>,
    #[ts(optional)]
    after: Option<Uuid>,
    #[ts(optional)]
    page_size: Option<u32>,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
struct QueryFeedbackBoundsByTargetIdParams {
    target_id: Uuid,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
struct CountFeedbackByTargetIdParams {
    target_id: Uuid,
}
