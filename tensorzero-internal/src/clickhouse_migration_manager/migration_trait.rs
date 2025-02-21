use crate::error::Error;
use async_trait::async_trait;

#[async_trait]
pub trait Migration {
    async fn can_apply(&self) -> Result<(), Error>;
    async fn should_apply(&self) -> Result<bool, Error>;
    async fn apply(&self) -> Result<(), Error>;
    fn rollback_instructions(&self) -> String;
    async fn has_succeeded(&self) -> Result<bool, Error>;
}
