//! Implementation of `TensorZeroClient` for the TensorZero SDK `Client`.
//!
//! This extends the `tensorzero::Client` type with the `TensorZeroClient` trait,
//! handling both HTTP gateway and embedded gateway modes via the client's internal
//! mode switching.

use async_trait::async_trait;
use tensorzero::{
    Client, ClientExt, ClientInferenceParams, ClientMode, CreateDatapointRequest,
    CreateDatapointsFromInferenceRequestParams, CreateDatapointsResponse, DeleteDatapointsResponse,
    GetDatapointsResponse, GetInferencesResponse, InferenceOutput, InferenceResponse,
    ListDatapointsRequest, ListInferencesRequest, TensorZeroError, UpdateDatapointRequest,
    UpdateDatapointsResponse,
};
use tensorzero_core::config::snapshot::SnapshotHash;
use uuid::Uuid;

use super::{
    CreateEventRequest, CreateEventResponse, ListEventsParams, ListEventsResponse,
    ListSessionsParams, ListSessionsResponse, TensorZeroClient, TensorZeroClientError,
};

/// Implementation of `TensorZeroClient` for the TensorZero SDK `Client`.
#[async_trait]
impl TensorZeroClient for Client {
    async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceResponse, TensorZeroClientError> {
        match Client::inference(self, params).await? {
            InferenceOutput::NonStreaming(response) => Ok(response),
            InferenceOutput::Streaming(_) => Err(TensorZeroClientError::StreamingNotSupported),
        }
    }

    async fn create_autopilot_event(
        &self,
        session_id: Uuid,
        request: CreateEventRequest,
    ) -> Result<CreateEventResponse, TensorZeroClientError> {
        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                // HTTP mode: call the internal endpoint
                let url = http
                    .base_url
                    .join(&format!(
                        "internal/autopilot/v1/sessions/{session_id}/events"
                    ))
                    .map_err(|e: url::ParseError| {
                        TensorZeroClientError::Autopilot(
                            autopilot_client::AutopilotError::InvalidUrl(e),
                        )
                    })?;

                let response = http
                    .http_client
                    .post(url)
                    .json(&request)
                    .send()
                    .await
                    .map_err(|e| {
                        TensorZeroClientError::Autopilot(autopilot_client::AutopilotError::Request(
                            e,
                        ))
                    })?;

                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    return Err(TensorZeroClientError::Autopilot(
                        autopilot_client::AutopilotError::Http {
                            status_code: status,
                            message: text,
                        },
                    ));
                }

                response.json().await.map_err(|e| {
                    TensorZeroClientError::Autopilot(autopilot_client::AutopilotError::Request(e))
                })
            }
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => {
                // Embedded mode: call the core function directly
                let autopilot_client = gateway
                    .handle
                    .app_state
                    .autopilot_client
                    .as_ref()
                    .ok_or(TensorZeroClientError::AutopilotUnavailable)?;

                tensorzero_core::endpoints::internal::autopilot::create_event(
                    autopilot_client,
                    session_id,
                    request,
                )
                .await
                .map_err(|e| {
                    TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
                })
            }
        }
    }

    async fn list_autopilot_events(
        &self,
        session_id: Uuid,
        params: ListEventsParams,
    ) -> Result<ListEventsResponse, TensorZeroClientError> {
        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                let mut url = http
                    .base_url
                    .join(&format!(
                        "internal/autopilot/v1/sessions/{session_id}/events"
                    ))
                    .map_err(|e: url::ParseError| {
                        TensorZeroClientError::Autopilot(
                            autopilot_client::AutopilotError::InvalidUrl(e),
                        )
                    })?;

                // Add query params
                if let Some(limit) = params.limit {
                    url.query_pairs_mut()
                        .append_pair("limit", &limit.to_string());
                }
                if let Some(before) = params.before {
                    url.query_pairs_mut()
                        .append_pair("before", &before.to_string());
                }

                let response = http.http_client.get(url).send().await.map_err(|e| {
                    TensorZeroClientError::Autopilot(autopilot_client::AutopilotError::Request(e))
                })?;

                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    return Err(TensorZeroClientError::Autopilot(
                        autopilot_client::AutopilotError::Http {
                            status_code: status,
                            message: text,
                        },
                    ));
                }

                response.json().await.map_err(|e| {
                    TensorZeroClientError::Autopilot(autopilot_client::AutopilotError::Request(e))
                })
            }
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => {
                let autopilot_client = gateway
                    .handle
                    .app_state
                    .autopilot_client
                    .as_ref()
                    .ok_or(TensorZeroClientError::AutopilotUnavailable)?;

                tensorzero_core::endpoints::internal::autopilot::list_events(
                    autopilot_client,
                    session_id,
                    params,
                )
                .await
                .map_err(|e| {
                    TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
                })
            }
        }
    }

    async fn list_autopilot_sessions(
        &self,
        params: ListSessionsParams,
    ) -> Result<ListSessionsResponse, TensorZeroClientError> {
        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                let mut url = http
                    .base_url
                    .join("internal/autopilot/v1/sessions")
                    .map_err(|e: url::ParseError| {
                        TensorZeroClientError::Autopilot(
                            autopilot_client::AutopilotError::InvalidUrl(e),
                        )
                    })?;

                // Add query params
                if let Some(limit) = params.limit {
                    url.query_pairs_mut()
                        .append_pair("limit", &limit.to_string());
                }
                if let Some(offset) = params.offset {
                    url.query_pairs_mut()
                        .append_pair("offset", &offset.to_string());
                }

                let response = http.http_client.get(url).send().await.map_err(|e| {
                    TensorZeroClientError::Autopilot(autopilot_client::AutopilotError::Request(e))
                })?;

                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    return Err(TensorZeroClientError::Autopilot(
                        autopilot_client::AutopilotError::Http {
                            status_code: status,
                            message: text,
                        },
                    ));
                }

                response.json().await.map_err(|e| {
                    TensorZeroClientError::Autopilot(autopilot_client::AutopilotError::Request(e))
                })
            }
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => {
                let autopilot_client = gateway
                    .handle
                    .app_state
                    .autopilot_client
                    .as_ref()
                    .ok_or(TensorZeroClientError::AutopilotUnavailable)?;

                tensorzero_core::endpoints::internal::autopilot::list_sessions(
                    autopilot_client,
                    params,
                )
                .await
                .map_err(|e| {
                    TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
                })
            }
        }
    }

    async fn action(
        &self,
        snapshot_hash: SnapshotHash,
        params: ClientInferenceParams,
    ) -> Result<InferenceResponse, TensorZeroClientError> {
        use tensorzero_core::endpoints::internal::action::{ActionInput, ActionInputInfo};

        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                let url = http
                    .base_url
                    .join("internal/action")
                    .map_err(|e: url::ParseError| {
                        TensorZeroClientError::Autopilot(
                            autopilot_client::AutopilotError::InvalidUrl(e),
                        )
                    })?;

                let action_input = ActionInputInfo {
                    snapshot_hash,
                    input: ActionInput::Inference(Box::new(params)),
                };

                let response = http
                    .http_client
                    .post(url)
                    .json(&action_input)
                    .send()
                    .await
                    .map_err(|e| {
                        TensorZeroClientError::Autopilot(autopilot_client::AutopilotError::Request(
                            e,
                        ))
                    })?;

                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    return Err(TensorZeroClientError::Autopilot(
                        autopilot_client::AutopilotError::Http {
                            status_code: status,
                            message: text,
                        },
                    ));
                }

                response.json().await.map_err(|e| {
                    TensorZeroClientError::Autopilot(autopilot_client::AutopilotError::Request(e))
                })
            }
            ClientMode::EmbeddedGateway {
                gateway,
                timeout: _,
            } => {
                let action_input = ActionInputInfo {
                    snapshot_hash,
                    input: ActionInput::Inference(Box::new(params)),
                };

                let result = tensorzero_core::endpoints::internal::action::action(
                    &gateway.handle.app_state,
                    action_input,
                )
                .await
                .map_err(|e| {
                    TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
                })?;

                match result {
                    tensorzero_core::endpoints::internal::action::ActionResponse::Inference(
                        response,
                    ) => Ok(response),
                    tensorzero_core::endpoints::internal::action::ActionResponse::Feedback(_) => {
                        Err(TensorZeroClientError::TensorZero(TensorZeroError::Other {
                            source: tensorzero_core::error::Error::new(
                                tensorzero_core::error::ErrorDetails::InternalError {
                                    message: "Unexpected feedback response from action endpoint"
                                        .to_string(),
                                },
                            )
                            .into(),
                        }))
                    }
                }
            }
        }
    }

    // ========== Datapoint CRUD Operations ==========

    async fn create_datapoints(
        &self,
        dataset_name: String,
        datapoints: Vec<CreateDatapointRequest>,
    ) -> Result<CreateDatapointsResponse, TensorZeroClientError> {
        ClientExt::create_datapoints(self, dataset_name, datapoints)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    async fn create_datapoints_from_inferences(
        &self,
        dataset_name: String,
        params: CreateDatapointsFromInferenceRequestParams,
    ) -> Result<CreateDatapointsResponse, TensorZeroClientError> {
        ClientExt::create_datapoints_from_inferences(self, dataset_name, params)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    async fn list_datapoints(
        &self,
        dataset_name: String,
        request: ListDatapointsRequest,
    ) -> Result<GetDatapointsResponse, TensorZeroClientError> {
        ClientExt::list_datapoints(self, dataset_name, request)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    async fn get_datapoints(
        &self,
        dataset_name: Option<String>,
        ids: Vec<Uuid>,
    ) -> Result<GetDatapointsResponse, TensorZeroClientError> {
        ClientExt::get_datapoints(self, dataset_name, ids)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    async fn update_datapoints(
        &self,
        dataset_name: String,
        datapoints: Vec<UpdateDatapointRequest>,
    ) -> Result<UpdateDatapointsResponse, TensorZeroClientError> {
        ClientExt::update_datapoints(self, dataset_name, datapoints)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    async fn delete_datapoints(
        &self,
        dataset_name: String,
        ids: Vec<Uuid>,
    ) -> Result<DeleteDatapointsResponse, TensorZeroClientError> {
        ClientExt::delete_datapoints(self, dataset_name, ids)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }

    // ========== Inference Query Operations ==========

    async fn list_inferences(
        &self,
        request: ListInferencesRequest,
    ) -> Result<GetInferencesResponse, TensorZeroClientError> {
        ClientExt::list_inferences(self, request)
            .await
            .map_err(TensorZeroClientError::TensorZero)
    }
}
