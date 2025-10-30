use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

#[async_trait]
pub trait Migration {
    // This needs to be on the trait itself (rather than a standalone function),
    // so that `&self` is the underlying type. This ensures that
    // calling `name()` on a `dyn Migration` will get the name of the erased type.
    fn name(&self) -> String {
        let name = std::any::type_name_of_val(&self)
            .split("::")
            .last()
            .unwrap_or("Unknown migration");
        // Remove the any lifetime parameter from the struct name (added in Rust 1.91)
        let name = name.strip_suffix("<'_>").unwrap_or(name);
        name.to_string()
    }
    fn migration_num(&self) -> Result<u32, Error> {
        let name = self.name();
        let id = name
            .strip_prefix("Migration")
            .ok_or_else(|| {
                Error::new(ErrorDetails::ClickHouseMigration {
                    id: name.clone(),
                    message: "Migration name does not start with 'Migration'".to_string(),
                })
            })?
            .parse::<u32>()
            .map_err(|e| {
                Error::new(ErrorDetails::ClickHouseMigration {
                    id: name,
                    message: format!("Migration has invalid numeric suffix: {e}"),
                })
            })?;
        Ok(id)
    }
    async fn can_apply(&self) -> Result<(), Error>;
    async fn should_apply(&self) -> Result<bool, Error>;
    async fn apply(&self, clean_start: bool) -> Result<(), Error>;
    /// ClickHouse queries that can be used to rollback the migration.
    /// Note - we run this as part of CI, so comments should not be on their own lines
    /// (as a comment is not a valid query by itself).
    fn rollback_instructions(&self) -> String;
    async fn has_succeeded(&self) -> Result<bool, Error>;
}
