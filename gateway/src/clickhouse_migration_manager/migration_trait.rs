use futures::Future;

use crate::error::Error;

pub(super) trait Migration {
    fn can_apply<'a>(&'a self) -> impl Future<Output = Result<(), Error>> + Send + 'a;
    fn should_apply<'a>(&'a self) -> impl Future<Output = Result<bool, Error>> + Send + 'a;
    fn apply<'a>(&'a self) -> impl Future<Output = Result<(), Error>> + Send + 'a;
    fn rollback<'a>(&'a self) -> impl Future<Output = Result<(), Error>> + Send + 'a;
    fn has_succeeded<'a>(&'a self) -> impl Future<Output = Result<bool, Error>> + Send + 'a;
}
