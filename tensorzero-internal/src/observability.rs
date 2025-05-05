use clap::ValueEnum;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanExporter};
use opentelemetry_sdk::Resource;
use tracing_subscriber::{filter, Registry};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};

use crate::error::{Error, ErrorDetails};

#[derive(Clone, Debug, Default, ValueEnum)]
pub enum LogFormat {
    #[default]
    Pretty,
    Json,
}

// Builds the internal OpenTelemetry layer, without any filtering applied.
fn internal_build_otel_layer<T: SpanExporter + 'static>(
    override_exporter: Option<T>,
) -> Result<impl Layer<Registry>, Error> {
    let mut provider = SdkTracerProvider::builder().with_resource(
        Resource::builder_empty()
            .with_attribute(KeyValue::new(
                opentelemetry_semantic_conventions::resource::SERVICE_NAME,
                "tensorzero-gateway",
            ))
            .build(),
    );

    if let Some(exporter) = override_exporter {
        provider = provider.with_simple_exporter(exporter)
    } else {
        provider = provider.with_batch_exporter(
            opentelemetry_otlp::SpanExporter::builder()
                .with_tonic()
                .build()
                .map_err(|e| {
                    Error::new(ErrorDetails::Observability {
                        message: format!("Failed to create OTLP exporter: {e}"),
                    })
                })?,
        );
    }
    let provider = provider.build();
    let tracer = provider.tracer("tensorzero");
    Ok(tracing_opentelemetry::layer().with_tracer(tracer))
}

/// Creates an OpenTelemetry export layer. This layer is disabled by default,
/// and can be dynamically enabled using the returned `DelayedOtelEnableHandle`.
/// See the `DelayedOtelEnableHandle` docs for more details.
///
/// The `override_exporter` parameter can be used to prove a custom `SpanExporter`.
/// This is used by `install_capturing_otel_exporter` during e2e tests to capture
/// all emitted spans.
///
/// If `override_exporter` is `None`, the default OTLP exporter will be used,
/// which is configured via environment variables (e..g `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`):
/// https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/protocol/exporter.md#endpoint-urls-for-otlphttp
pub fn build_opentelemetry_layer<T: SpanExporter + 'static>(
    override_exporter: Option<T>,
) -> Result<(DelayedOtelEnableHandle, impl Layer<Registry>), Error> {
    let (otel_reload_layer, reload_handle) = tracing_subscriber::reload::Layer::new(None);

    // Avoid exposing all of our internal spans, as we don't want customers to start depending on them.
    // We only expose spans that explicitly contain field prefixed with "http." or "otel."
    // For example, `#[instrument(fields(otel.name = "my_otel_name"))]` will be exported
    let filter = filter::filter_fn(|metadata| {
        metadata
            .fields()
            .iter()
            .any(|field| field.name().starts_with("http.") || field.name().starts_with("otel."))
    });

    let delayed_enable = DelayedOtelEnableHandle {
        enable_cb: Box::new(move || {
            let base_otel_layer = internal_build_otel_layer(override_exporter)?;
            reload_handle
                .modify(|l| {
                    *l = Some(base_otel_layer);
                })
                .map_err(|e| {
                    Error::new(ErrorDetails::Observability {
                        message: format!("Failed to enable OTLP exporter: {e}"),
                    })
                })?;
            Ok(())
        }),
    };
    // Note - we apply our filter on top of the reloadable layer.
    // We cannot apply the filter inside of the reloadable layer (e.g. directly on the otel layer),
    // as this will produce a panic due to https://github.com/tokio-rs/tracing/issues/1629
    // This is slightly less efficient when OTEL is disabled (since we'll apply a filter and then
    // invoke a no-op layer), but it shouldn't matter in practice.
    Ok((delayed_enable, otel_reload_layer.with_filter(filter)))
}

/// A handle produced by `build_opentelemetry_layer` to allow enabling the OTEL layer
/// after tracing as been initialized.
/// Background: During gateway initialization, we need to:
/// * Set up the global tracing subscriber
/// * Log some startup info (e.g. version, git hash)
/// * Try to load and parse the config file from disk
///
/// The config file is responsible for controlling whether OTEL is enabled,
/// but we want to use `tracing` before and during config file parsing.
///
/// The solution is to use `tracing_subscriber::reload` to create a reloadable layer.
/// a wrapped layer, which can later be enabled based on the config file value.
/// The gateway unconditionally registers the layer returned by `build_opentelemetry_layer`,
/// and later determines whether to call `enable_otel` based on the config file.
pub struct DelayedOtelEnableHandle {
    enable_cb: Box<dyn FnOnce() -> Result<(), Error> + Send + Sync>,
}

impl DelayedOtelEnableHandle {
    pub fn enable_otel(self) -> Result<(), Error> {
        (self.enable_cb)()
    }
}

/// Set up logs
pub fn setup_logs(debug: bool, log_format: LogFormat) -> Result<DelayedOtelEnableHandle, Error> {
    let default_level = if debug { "debug,warn" } else { "warn" };
    // Get the current log level from the environment variable `RUST_LOG`
    let log_level = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        format!("warn,gateway={default_level},tensorzero_internal={default_level}").into()
    });

    let log_layer = match log_format {
        LogFormat::Pretty => {
            Box::new(tracing_subscriber::fmt::layer()) as Box<dyn Layer<_> + Send + Sync>
        }
        LogFormat::Json => Box::new(tracing_subscriber::fmt::layer().json()),
    };

    // We need to provide a dummy generic parameter to satisfy the compiler
    let (delayed_enable, otel_layer) =
        build_opentelemetry_layer::<opentelemetry_otlp::SpanExporter>(None)?;

    tracing_subscriber::registry()
        .with(otel_layer)
        .with(log_layer.with_filter(log_level))
        .init();
    Ok(delayed_enable)
}

/// Set up Prometheus metrics exporter
pub fn setup_metrics() -> Result<PrometheusHandle, Error> {
    PrometheusBuilder::new().install_recorder().map_err(|e| {
        Error::new(ErrorDetails::Observability {
            message: format!("Failed to install Prometheus exporter: {e}"),
        })
    })
}
