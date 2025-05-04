use async_trait::async_trait;

use super::check_table_exists;
use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};

/// This migration adds the `DynamicEvaluationRunByProjectName` table and the
/// `DynamicEvaluationRunByProjectNameView` materialized view.
/// These support consumption of dynamic evaluations indexed by project name.
/// The `DynamicEvaluationRunByProjectName` table contains the same data as the
/// `DynamicEvaluationRun` table with different indexing.
pub struct Migration0027<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0027<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        let dynamic_evaluation_run_table_exists =
            check_table_exists(self.clickhouse, "DynamicEvaluationRun", "0027").await?;
        if !dynamic_evaluation_run_table_exists {
            return Err(ErrorDetails::ClickHouseMigration {
                id: "0027".to_string(),
                message: "DynamicEvaluationRun table does not exist".to_string(),
            }
            .into());
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let dynamic_evaluation_run_by_project_name_table_exists =
            check_table_exists(self.clickhouse, "DynamicEvaluationRunByProjectName", "0027")
                .await?;
        let dynamic_evaluation_run_by_project_name_view_exists = check_table_exists(
            self.clickhouse,
            "DynamicEvaluationRunByProjectNameView",
            "0027",
        )
        .await?;

        Ok(!dynamic_evaluation_run_by_project_name_table_exists
            || !dynamic_evaluation_run_by_project_name_view_exists)
    }

    async fn apply(&self) -> Result<(), Error> {
        let query = r#"
            CREATE TABLE IF NOT EXISTS DynamicEvaluationRunByProjectName
                (
                    run_id_uint UInt128, -- UUID encoded as a UInt128
                    variant_pins Map(String, String),
                    tags Map(String, String),
                    project_name String,
                    run_display_name Nullable(String),
                    is_deleted Bool DEFAULT false,
                    updated_at DateTime64(6, 'UTC') DEFAULT now()
                ) ENGINE = ReplacingMergeTree(updated_at, is_deleted)
                ORDER BY (project_name, run_id_uint);
        "#;
        let _ = self
            .clickhouse
            .run_query_synchronous(query.to_string(), None)
            .await?;

        let query = r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS DynamicEvaluationRunByProjectNameView
                TO DynamicEvaluationRunByProjectName
                AS
                SELECT * FROM DynamicEvaluationRun
                WHERE project_name IS NOT NULL
                ORDER BY project_name, run_id_uint;
        "#;
        let _ = self
            .clickhouse
            .run_query_synchronous(query.to_string(), None)
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "\
        -- Drop the materialized view\n\
        DROP MATERIALIZED VIEW IF EXISTS DynamicEvaluationRunByProjectNameView;\n\
        -- Drop the `DynamicEvaluationRunByProjectName` table\n\
        DROP TABLE IF EXISTS DynamicEvaluationRunByProjectName;
        "
        .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
