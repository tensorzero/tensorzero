//! Code is based on <https://github.com/hatoo/http-mitm-proxy/blob/7c8c3bde77173af6385d5d0ffaea6105498df1ff/src/lib.rs> (MIT-licensed)
#![expect(clippy::unwrap_used)]

use crate::tls::{generate_cert, CertifiedKeyDer};
use http_body_util::{combinators::BoxBody, BodyExt, Empty};
use hyper::{
    body::{Body, Incoming},
    server,
    service::{service_fn, HttpService},
    Method, Request, Response, StatusCode,
};
use hyper_util::rt::{TokioExecutor, TokioIo};
use moka::sync::Cache;
use std::{borrow::Borrow, error::Error as StdError, future::Future, net::SocketAddr, sync::Arc};
use tokio::net::{TcpListener, TcpStream, ToSocketAddrs};
use tokio_rustls::rustls;

#[derive(Clone)]
/// The main struct to run proxy server
pub struct MitmProxy<C> {
    /// Root certificate to sign fake certificates. You may need to trust this certificate on client application to use HTTPS.
    ///
    /// If None, proxy will just tunnel HTTPS traffic and will not observe HTTPS traffic.
    pub root_cert: Option<C>,
    /// Cache to store generated certificates. If None, cache will not be used.
    /// If `root_cert` is None, cache will not be used.
    ///
    /// The key of cache is hostname.
    pub cert_cache: Option<Cache<String, CertifiedKeyDer>>,
}

impl<C> MitmProxy<C> {
    /// Create a new `MitmProxy`
    pub fn new(root_cert: Option<C>, cache: Option<Cache<String, CertifiedKeyDer>>) -> Self {
        Self {
            root_cert,
            cert_cache: cache,
        }
    }
}

impl<C> MitmProxy<C>
where
    C: Borrow<rcgen::Issuer<'static, rcgen::KeyPair>> + Send + Sync + 'static,
{
    /// Bind to a socket address and return a future that runs the proxy server.
    /// URL for requests that passed to service are full URL including scheme.
    pub async fn bind<A: ToSocketAddrs, S>(
        self,
        addr: A,
        service: S,
    ) -> Result<(SocketAddr, impl Future<Output = ()>), std::io::Error>
    where
        S: HttpService<Incoming> + Clone + Send + 'static,
        S::Error: Into<Box<dyn StdError + Send + Sync>>,
        S::ResBody: Send + Sync + 'static,
        <S::ResBody as Body>::Data: Send,
        <S::ResBody as Body>::Error: Into<Box<dyn StdError + Send + Sync>>,
        S::Future: Send,
    {
        let listener = TcpListener::bind(addr).await?;

        let proxy = Arc::new(self);

        Ok((listener.local_addr()?, async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    continue;
                };

                let service = service.clone();

                let proxy = proxy.clone();
                tokio::spawn(async move {
                    if let Err(err) = server::conn::http1::Builder::new()
                        .preserve_header_case(true)
                        .title_case_headers(true)
                        .serve_connection(
                            TokioIo::new(stream),
                            Self::wrap_service(proxy.clone(), service.clone()),
                        )
                        .with_upgrades()
                        .await
                    {
                        tracing::error!("Error in proxy: {err:?}");
                    }
                });
            }
        }))
    }

    /// Transform a service to a service that can be used in hyper server.
    /// URL for requests that passed to service are full URL including scheme.
    pub fn wrap_service<S>(
        proxy: Arc<Self>,
        service: S,
    ) -> impl HttpService<
        Incoming,
        ResBody = BoxBody<<S::ResBody as Body>::Data, <S::ResBody as Body>::Error>,
        Future: Send,
    >
    where
        S: HttpService<Incoming> + Clone + Send + 'static,
        S::Error: Into<Box<dyn StdError + Send + Sync>>,
        S::ResBody: Send + Sync + 'static,
        <S::ResBody as Body>::Data: Send,
        <S::ResBody as Body>::Error: Into<Box<dyn StdError + Send + Sync>>,
        S::Future: Send,
    {
        service_fn(move |req| {
            let proxy = proxy.clone();
            let mut service = service.clone();

            async move {
                if req.method() == Method::CONNECT {
                    // https
                    let Some(connect_authority) = req.uri().authority().cloned() else {
                        tracing::error!(
                            "Bad CONNECT request: {}, Reason: Invalid Authority",
                            req.uri()
                        );
                        return Ok(no_body(StatusCode::BAD_REQUEST)
                            .map(|b| b.boxed().map_err(|never| match never {}).boxed()));
                    };

                    tokio::spawn(async move {
                        let Ok(client) = hyper::upgrade::on(req).await else {
                            tracing::error!(
                                "Bad CONNECT request: {}, Reason: Invalid Upgrade",
                                connect_authority
                            );
                            return;
                        };
                        if let Some(server_config) =
                            proxy.server_config(connect_authority.host().to_string(), true)
                        {
                            let server_config = match server_config {
                                Ok(server_config) => server_config,
                                Err(err) => {
                                    tracing::error!(
                                        "Failed to create server config for {}, {}",
                                        connect_authority.host(),
                                        err
                                    );
                                    return;
                                }
                            };
                            let server_config = Arc::new(server_config);
                            let tls_acceptor = tokio_rustls::TlsAcceptor::from(server_config);
                            let client = match tls_acceptor.accept(TokioIo::new(client)).await {
                                Ok(client) => client,
                                Err(err) => {
                                    tracing::error!(
                                        "Failed to accept TLS connection for {}, {}",
                                        connect_authority.host(),
                                        err
                                    );
                                    return;
                                }
                            };
                            let f = move |mut req: Request<_>| {
                                let connect_authority = connect_authority.clone();
                                let mut service = service.clone();

                                async move {
                                    inject_authority(&mut req, connect_authority.clone());
                                    service.call(req).await
                                }
                            };
                            let res = if client.get_ref().1.alpn_protocol() == Some(b"h2") {
                                server::conn::http2::Builder::new(TokioExecutor::new())
                                    .serve_connection(TokioIo::new(client), service_fn(f))
                                    .await
                            } else {
                                server::conn::http1::Builder::new()
                                    .preserve_header_case(true)
                                    .title_case_headers(true)
                                    .serve_connection(TokioIo::new(client), service_fn(f))
                                    .with_upgrades()
                                    .await
                            };

                            if let Err(_err) = res {
                                // Suppress error because if we serving HTTPS proxy server and forward to HTTPS server, it will always error when closing connection.
                                // tracing::error!("Error in proxy: {}", err);
                            }
                        } else {
                            let Ok(mut server) =
                                TcpStream::connect(connect_authority.as_str()).await
                            else {
                                tracing::error!("Failed to connect to {}", connect_authority);
                                return;
                            };
                            let _ = tokio::io::copy_bidirectional(
                                &mut TokioIo::new(client),
                                &mut server,
                            )
                            .await;
                        }
                    });

                    Ok(Response::new(
                        http_body_util::Empty::new()
                            .map_err(|never: std::convert::Infallible| match never {})
                            .boxed(),
                    ))
                } else {
                    // http
                    service.call(req).await.map(|res| res.map(BodyExt::boxed))
                }
            }
        })
    }

    fn get_certified_key(&self, host: String) -> Option<CertifiedKeyDer> {
        self.root_cert.as_ref().map(|root_cert| {
            if let Some(cache) = self.cert_cache.as_ref() {
                cache.get_with(host.clone(), move || {
                    generate_cert(host, root_cert.borrow())
                })
            } else {
                generate_cert(host, root_cert.borrow())
            }
        })
    }

    fn server_config(
        &self,
        host: String,
        h2: bool,
    ) -> Option<Result<rustls::ServerConfig, rustls::Error>> {
        if let Some(cert) = self.get_certified_key(host) {
            let config = rustls::ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(
                    vec![rustls::pki_types::CertificateDer::from(cert.cert_der)],
                    rustls::pki_types::PrivateKeyDer::Pkcs8(
                        rustls::pki_types::PrivatePkcs8KeyDer::from(cert.key_der),
                    ),
                );

            Some(if h2 {
                config.map(|mut server_config| {
                    server_config.alpn_protocols = vec!["h2".into(), "http/1.1".into()];
                    server_config
                })
            } else {
                config
            })
        } else {
            None
        }
    }
}

fn no_body<D>(status: StatusCode) -> Response<Empty<D>> {
    let mut res = Response::new(Empty::new());
    *res.status_mut() = status;
    res
}

fn inject_authority<B>(request_middleman: &mut Request<B>, authority: hyper::http::uri::Authority) {
    let mut parts = request_middleman.uri().clone().into_parts();
    parts.scheme = Some(hyper::http::uri::Scheme::HTTPS);
    if parts.authority.is_none() {
        parts.authority = Some(authority);
    }
    *request_middleman.uri_mut() = hyper::http::uri::Uri::from_parts(parts).unwrap();
}
