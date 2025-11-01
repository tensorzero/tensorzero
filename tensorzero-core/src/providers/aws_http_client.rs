// Based on https://github.com/aws/amazon-q-developer-cli/blob/858f9417dcc131e140dfc6cce5a4b657af56616a/crates/fig_aws_common/src/http_client.rs
// (MIT-licensed)

use std::time::Duration;

use aws_smithy_runtime_api::client::http::{
    HttpClient, HttpConnector, HttpConnectorFuture, HttpConnectorSettings, SharedHttpConnector,
};
use aws_smithy_runtime_api::client::result::ConnectorError;
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_runtime_api::http::Request;
use aws_smithy_types::body::SdkBody;

use crate::http::TensorzeroHttpClient;

/// A wrapper around [TensorzeroHttpClient] that implements [HttpClient].
///
/// This is required to support using proxy servers with the AWS SDK.
#[derive(Debug, Clone)]
pub struct Client {
    inner: TensorzeroHttpClient,
}

impl Client {
    pub fn new(client: TensorzeroHttpClient) -> Self {
        Self { inner: client }
    }
}

#[derive(Debug)]
struct CallError {
    kind: CallErrorKind,
    message: &'static str,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl CallError {
    fn user(message: &'static str) -> Self {
        Self {
            kind: CallErrorKind::User,
            message,
            source: None,
        }
    }

    fn user_with_source<E>(message: &'static str, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            kind: CallErrorKind::User,
            message,
            source: Some(Box::new(source)),
        }
    }

    fn timeout<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            kind: CallErrorKind::Timeout,
            message: "request timed out",
            source: Some(Box::new(source)),
        }
    }

    fn io<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            kind: CallErrorKind::Io,
            message: "an i/o error occurred",
            source: Some(Box::new(source)),
        }
    }

    fn other<E>(message: &'static str, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            kind: CallErrorKind::Other,
            message,
            source: Some(Box::new(source)),
        }
    }
}

impl std::error::Error for CallError {}

impl std::fmt::Display for CallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(err) = self.source.as_ref() {
            write!(f, ": {err}")?;
        }
        Ok(())
    }
}

impl From<CallError> for ConnectorError {
    fn from(value: CallError) -> Self {
        match &value.kind {
            CallErrorKind::User => Self::user(Box::new(value)),
            CallErrorKind::Timeout => Self::timeout(Box::new(value)),
            CallErrorKind::Io => Self::io(Box::new(value)),
            CallErrorKind::Other => Self::other(Box::new(value), None),
        }
    }
}

impl From<reqwest::Error> for CallError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            CallError::timeout(err)
        } else if err.is_connect() {
            CallError::io(err)
        } else {
            CallError::other("an unknown error occurred", err)
        }
    }
}

#[derive(Debug, Clone)]
enum CallErrorKind {
    User,
    Timeout,
    Io,
    Other,
}

#[derive(Debug)]
struct ReqwestConnector {
    client: TensorzeroHttpClient,
    timeout: Option<Duration>,
}

impl HttpConnector for ReqwestConnector {
    fn call(&self, request: Request) -> HttpConnectorFuture {
        let client = self.client.clone();
        let timeout = self.timeout;

        HttpConnectorFuture::new(async move {
            // Convert the aws_smithy_runtime_api request to a reqwest request.
            // TODO: There surely has to be a better way to convert an aws_smith_runtime_api
            // Request<SdkBody> to a reqwest Request<Body>.
            let mut req_builder = client.request(
                reqwest::Method::from_bytes(request.method().as_bytes()).map_err(|err| {
                    CallError::user_with_source("failed to create method name", err)
                })?,
                request.uri().to_owned(),
            );
            // Copy the header, body, and timeout.
            let parts = request.into_parts();
            for (name, value) in &parts.headers {
                let name = name.to_owned();
                let value = value.as_bytes().to_owned();
                req_builder = req_builder.header(name, value);
            }
            let body_bytes = parts
                .body
                .bytes()
                .ok_or_else(|| CallError::user("streaming request body is not supported"))?
                .to_owned();
            req_builder = req_builder.body(body_bytes);
            if let Some(timeout) = timeout {
                req_builder = req_builder.timeout(timeout);
            }

            let reqwest_response = req_builder.send().await.map_err(CallError::from)?;

            // Converts from a reqwest Response into an http::Response<SdkBody>.
            let (parts, body) = http::Response::from(reqwest_response).into_parts();
            let http_response = http::Response::from_parts(parts, SdkBody::from_body_1_x(body));

            Ok(
                aws_smithy_runtime_api::http::Response::try_from(http_response).map_err(|err| {
                    CallError::other("failed to convert to a proper response", err)
                })?,
            )
        })
    }
}

impl HttpClient for Client {
    fn http_connector(
        &self,
        settings: &HttpConnectorSettings,
        _components: &RuntimeComponents,
    ) -> SharedHttpConnector {
        let connector = ReqwestConnector {
            client: self.inner.clone(),
            timeout: settings.read_timeout(),
        };
        SharedHttpConnector::new(connector)
    }
}
