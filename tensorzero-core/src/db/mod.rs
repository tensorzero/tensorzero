use std::future::Future;

use crate::error::Error;

pub mod clickhouse;

pub trait DatabaseConnection {
    fn health(&self) -> impl Future<Output = Result<(), Error>>;
}
