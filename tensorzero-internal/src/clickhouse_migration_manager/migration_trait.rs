use crate::error::Error;
use async_trait::async_trait;

#[async_trait]
pub trait Migration {
    // This needs to be on the trait itself (rather than a standalone function),
    // so that `&self` is the underlying type. This ensures that
    // calling `name()` on a `dyn Migration` will get the name of the erased type.
    fn name(&self) -> String {
        std::any::type_name_of_val(&self)
            .split("::")
            .last()
            .unwrap_or("Unknown migration")
            .to_string()
    }
    async fn can_apply(&self) -> Result<(), Error>;
    async fn should_apply(&self) -> Result<bool, Error>;
    async fn apply(&self) -> Result<(), Error>;
    fn rollback_instructions(&self) -> String;
    async fn has_succeeded(&self) -> Result<bool, Error>;
}
