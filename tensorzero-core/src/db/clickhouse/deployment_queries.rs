use async_trait::async_trait;

use super::ClickHouseConnectionInfo;
use crate::db::DeploymentIdQueries;
use crate::error::{Error, ErrorDetails};

#[async_trait]
impl DeploymentIdQueries for ClickHouseConnectionInfo {
    /// Gets the deployment ID from ClickHouse.
    /// It is stored in the `DeploymentID` table as a 64 char hex hash.
    async fn get_deployment_id(&self) -> Result<String, Error> {
        let response = self
            .run_query_synchronous_no_params(
                "SELECT deployment_id FROM DeploymentID LIMIT 1".to_string(),
            )
            .await?;
        if response.response.is_empty() {
            return Err(Error::new(ErrorDetails::ClickHouseQuery {
                message: "Failed to get deployment ID from ClickHouse (response was empty)"
                    .to_string(),
            }));
        }
        Ok(response.response.trim().to_string())
    }
}
