use serde::Deserialize;
use std::sync::Arc;
use tensorzero::{
    ClickHouseConnection, CountDatapointsForDatasetFunctionParams, DatasetQueryParams,
    GetAdjacentDatapointIdsParams, GetDatapointParams, GetDatasetMetadataParams,
    StaleDatapointParams, TimeWindow,
};
use tensorzero_core::config::{Config, ConfigFileGlob};
use tensorzero_core::db::datasets::GetDatapointsParams;
use tensorzero_core::endpoints::datasets::v1::types::{
    GetDatapointsResponse, ListDatapointsRequest,
};
use tensorzero_core::endpoints::datasets::Datapoint;
use tensorzero_core::utils::gateway::setup_clickhouse;
use uuid::Uuid;

#[napi(js_name = "DatabaseClient")]
pub struct DatabaseClient {
    connection: Box<dyn ClickHouseConnection>,
    config: Arc<Config>,
}

#[napi]
impl DatabaseClient {
    #[napi(factory)]
    pub async fn from_clickhouse_url(
        clickhouse_url: String,
        config_path: Option<String>,
    ) -> Result<Self, napi::Error> {
        // Load config from the provided path or use empty config if None
        let config = if let Some(path) = config_path {
            let config_glob =
                ConfigFileGlob::new(path).map_err(|e| napi::Error::from_reason(e.to_string()))?;
            Arc::new(
                Config::load_and_verify_from_path(&config_glob)
                    .await
                    .map_err(|e| napi::Error::from_reason(e.to_string()))?,
            )
        } else {
            Arc::new(
                Config::new_empty()
                    .await
                    .map_err(|e| napi::Error::from_reason(e.to_string()))?,
            )
        };

        // Setup ClickHouse with the loaded config
        let connection = setup_clickhouse(&config, Some(clickhouse_url), true)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;

        Ok(Self {
            connection: Box::new(connection),
            config,
        })
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
    pub async fn count_rows_for_dataset(&self, params: String) -> Result<u32, napi::Error> {
        napi_call_no_deserializing!(&self, count_rows_for_dataset, params, DatasetQueryParams)
    }

    #[napi]
    pub async fn insert_rows_for_dataset(&self, params: String) -> Result<u32, napi::Error> {
        napi_call_no_deserializing!(&self, insert_rows_for_dataset, params, DatasetQueryParams)
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

    #[napi]
    pub async fn list_datapoints(
        &self,
        dataset_name: String,
        params: String,
    ) -> Result<String, napi::Error> {
        // Deserialize ListDatapointsRequest from the params string
        let request: ListDatapointsRequest =
            serde_json::from_str(&params).map_err(|e| napi::Error::from_reason(e.to_string()))?;

        // Convert to GetDatapointsParams
        let get_params = GetDatapointsParams {
            dataset_name: Some(dataset_name),
            function_name: request.function_name,
            ids: None,
            page_size: request.page_size.unwrap_or(20),
            offset: request.offset.unwrap_or(0),
            allow_stale: false,
            filter: request.filter,
        };

        // Call get_datapoints on the database connection
        let stored_datapoints = self
            .connection
            .get_datapoints(&get_params)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;

        // Convert StoredDatapoint → Datapoint using config
        let datapoints: Result<Vec<Datapoint>, _> = stored_datapoints
            .into_iter()
            .map(|dp| dp.into_datapoint(&self.config))
            .collect();
        let datapoints = datapoints.map_err(|e| napi::Error::from_reason(e.to_string()))?;

        // Wrap in GetDatapointsResponse structure
        let response = GetDatapointsResponse { datapoints };

        // Serialize and return the result
        serde_json::to_string(&response).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub async fn get_feedback_by_variant(&self, params: String) -> Result<String, napi::Error> {
        let params_struct: GetFeedbackByVariantParams =
            serde_json::from_str(&params).map_err(|e| napi::Error::from_reason(e.to_string()))?;

        let result = self
            .connection
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
    #[ts(optional)]
    pub before: Option<Uuid>,
    #[ts(optional)]
    pub after: Option<Uuid>,
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
struct QueryFeedbackByTargetIdParams {
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    page_size: Option<u32>,
}

#[derive(Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
struct QueryDemonstrationFeedbackByInferenceIdParams {
    inference_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
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

#[derive(Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
struct GetFeedbackByVariantParams {
    metric_name: String,
    function_name: String,
    #[ts(optional)]
    variant_names: Option<Vec<String>>,
}
