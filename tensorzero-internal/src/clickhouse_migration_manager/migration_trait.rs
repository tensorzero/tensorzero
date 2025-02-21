use std::pin::Pin;

use futures::Future;

use crate::error::Error;

pub trait Migration {
    fn can_apply(&self) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;
    fn should_apply(&self) -> Pin<Box<dyn Future<Output = Result<bool, Error>> + Send + '_>>;
    fn apply(&self) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;
    fn rollback_instructions(&self) -> String;
    fn has_succeeded(&self) -> Pin<Box<dyn Future<Output = Result<bool, Error>> + Send + '_>>;
}
