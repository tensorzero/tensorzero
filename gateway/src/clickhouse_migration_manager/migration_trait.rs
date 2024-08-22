use std::any::Any;

use async_trait::async_trait;

use crate::error::Error;

#[async_trait]
pub trait Migration: Any {
    async fn can_apply(&self) -> Result<(), Error>;
    async fn should_apply(&self) -> Result<bool, Error>;
    async fn apply(&self) -> Result<(), Error>;
    fn rollback_instructions(&self) -> String;
    async fn has_succeeded(&self) -> Result<bool, Error>;
}
