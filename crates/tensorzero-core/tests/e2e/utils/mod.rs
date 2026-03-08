pub mod poll_for_result;

/// Skips the current test if the observability backend is not ClickHouse.
///
/// Usage: `skip_for_postgres!();`
macro_rules! skip_for_postgres {
    () => {
        if tensorzero_core::db::delegating_connection::PrimaryDatastore::from_test_env()
            != tensorzero_core::db::delegating_connection::PrimaryDatastore::ClickHouse
        {
            return;
        }
    };
}

pub(crate) use skip_for_postgres;
