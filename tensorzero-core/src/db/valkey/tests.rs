use crate::error::ErrorDetails;
use crate::observability::{LogFormat, setup_observability};

use super::ValkeyConnectionInfo;

/// Test that connecting with a TLS URL (`rediss://`) produces a connection error,
/// not a rustls crypto provider panic. This validates that `setup_observability`
/// correctly installs the rustls crypto provider so that the `redis` crate's
/// TLS support works.
#[tokio::test]
async fn test_tls_url_gives_connection_error() {
    setup_observability(LogFormat::Pretty, false).await.unwrap();

    let result = ValkeyConnectionInfo::new("rediss://tensorzero.invalid:6379").await;
    let err = result
        .err()
        .expect("TLS connection to non-TLS server should fail");
    assert!(
        matches!(err.get_details(), ErrorDetails::ValkeyConnection { .. }),
        "expected ValkeyConnection error, got: {err}"
    );
}

#[tokio::test]
async fn test_tls_url_gives_connection_error_cache_only() {
    setup_observability(LogFormat::Pretty, false).await.unwrap();

    let result = ValkeyConnectionInfo::new_cache_only("rediss://tensorzero.invalid:6379").await;
    let err = result
        .err()
        .expect("TLS connection to non-TLS server should fail");
    assert!(
        matches!(err.get_details(), ErrorDetails::ValkeyConnection { .. }),
        "expected ValkeyConnection error, got: {err}"
    );
}
