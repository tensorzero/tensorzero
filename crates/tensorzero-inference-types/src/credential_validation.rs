use std::future::Future;

tokio::task_local! {
    /// When set, we skip performing credential validation in model providers.
    /// This is used when running in e2e test mode, and by the 'evaluations' binary.
    /// We need to access this from async code (e.g. when looking up GCP SDK credentials),
    /// so this needs to be a tokio task-local (as a task may be moved between threads).
    ///
    /// Since this needs to be accessed from a `Deserialize` impl, it needs to
    /// be stored in a `static`, since we cannot pass in extra parameters when calling `Deserialize::deserialize`.
    static SKIP_CREDENTIAL_VALIDATION: ();
}

pub fn skip_credential_validation() -> bool {
    // tokio::task_local doesn't have an 'is_set' method, so we call 'try_with'
    // (which returns an `Err` if the task-local is not set)
    SKIP_CREDENTIAL_VALIDATION.try_with(|()| ()).is_ok()
}

/// Runs the provider future with credential validation disabled.
/// This is safe to repeatedly nest — the original credential validation
/// behavior will be restored after the outermost future completes.
pub async fn with_skip_credential_validation<T>(f: impl Future<Output = T>) -> T {
    SKIP_CREDENTIAL_VALIDATION.scope((), f).await
}

/// In e2e test mode, we skip credential validation by default.
/// This can be overridden by setting the `TENSORZERO_E2E_CREDENTIAL_VALIDATION` environment variable to `1`.
/// Outside of e2e test mode, we leave the behavior unchanged (other parts of the codebase might still
/// skip credential validation, e.g. when running in relay mode).
pub fn e2e_skip_credential_validation() -> bool {
    cfg!(any(test, feature = "e2e_tests"))
        && !std::env::var("TENSORZERO_E2E_CREDENTIAL_VALIDATION").is_ok_and(|x| x == "1")
}
