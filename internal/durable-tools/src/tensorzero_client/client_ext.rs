//! Implementation of `TensorZeroClient` for the TensorZero SDK `Client`.
//!
//! This extends the `tensorzero::Client` type with the `TensorZeroClient` trait,
//! handling both HTTP gateway and embedded gateway modes via the client's internal
//! mode switching.

use async_trait::async_trait;
use autopilot_client::AutopilotError;
use tensorzero::{
    Client, ClientInferenceParams, ClientMode, InferenceOutput, InferenceResponse, TensorZeroError,
};
use tensorzero_core::config::snapshot::SnapshotHash;
use tensorzero_core::endpoints::internal::action::{ActionInput, ActionInputInfo};
use tensorzero_core::endpoints::internal::autopilot::list_sessions;
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
                        TensorZeroClientError::Autopilot(AutopilotError::InvalidUrl(e))
                    })?;

                let response = http
                    .http_client
                    .post(url)
                    .json(&request)
                    .send()
                    .await
                    .map_err(|e| TensorZeroClientError::Autopilot(AutopilotError::Request(e)))?;

                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    return Err(TensorZeroClientError::Autopilot(AutopilotError::Http {
                        status_code: status,
                        message: text,
                    }));
                }

                response
                    .json()
                    .await
                    .map_err(|e| TensorZeroClientError::Autopilot(AutopilotError::Request(e)))
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
                        TensorZeroClientError::Autopilot(AutopilotError::InvalidUrl(e))
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

                let response =
                    http.http_client.get(url).send().await.map_err(|e| {
                        TensorZeroClientError::Autopilot(AutopilotError::Request(e))
                    })?;

                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    return Err(TensorZeroClientError::Autopilot(AutopilotError::Http {
                        status_code: status,
                        message: text,
                    }));
                }

                response
                    .json()
                    .await
                    .map_err(|e| TensorZeroClientError::Autopilot(AutopilotError::Request(e)))
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
                        TensorZeroClientError::Autopilot(AutopilotError::InvalidUrl(e))
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

                let response =
                    http.http_client.get(url).send().await.map_err(|e| {
                        TensorZeroClientError::Autopilot(AutopilotError::Request(e))
                    })?;

                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    return Err(TensorZeroClientError::Autopilot(AutopilotError::Http {
                        status_code: status,
                        message: text,
                    }));
                }

                response
                    .json()
                    .await
                    .map_err(|e| TensorZeroClientError::Autopilot(AutopilotError::Request(e)))
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

                list_sessions(autopilot_client, params).await.map_err(|e| {
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
        match self.mode() {
            ClientMode::HTTPGateway(http) => {
                let url = http
                    .base_url
                    .join("internal/action")
                    .map_err(|e: url::ParseError| {
                        TensorZeroClientError::Autopilot(AutopilotError::InvalidUrl(e))
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
                    .map_err(|e| TensorZeroClientError::Autopilot(AutopilotError::Request(e)))?;

                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    return Err(TensorZeroClientError::Autopilot(AutopilotError::Http {
                        status_code: status,
                        message: text,
                    }));
                }

                response
                    .json()
                    .await
                    .map_err(|e| TensorZeroClientError::Autopilot(AutopilotError::Request(e)))
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
}
