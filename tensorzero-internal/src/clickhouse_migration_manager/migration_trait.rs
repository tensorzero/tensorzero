use futures::Future;

use crate::error::Error;

pub trait Migration {
    fn can_apply(&self) -> impl Future<Output = Result<(), Error>> + Send;
    fn should_apply(&self) -> impl Future<Output = Result<bool, Error>> + Send;
    fn apply(&self) -> impl Future<Output = Result<(), Error>> + Send;
    fn rollback_instructions(&self) -> String;
    fn has_succeeded(&self) -> impl Future<Output = Result<bool, Error>> + Send;
}
