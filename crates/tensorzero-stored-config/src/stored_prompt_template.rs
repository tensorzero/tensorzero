use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Reference to a `prompt_template_versions_config` row.
/// Replaces `ResolvedTomlPathData` in all stored config types.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredPromptRef {
    pub prompt_template_version_id: Uuid,
    pub template_key: String,
}

/// A prompt template version stored in the database.
/// This is the stored equivalent of `ResolvedTomlPathData`, which eagerly loads
/// file contents from disk. The stored version keeps the template body inline
/// and tracks its identity via a UUID.
#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredPromptTemplate {
    pub id: Uuid,
    pub template_key: String,
    pub source_body: String,
    /// BLAKE3 hash of `source_body`, used to deduplicate identical template content.
    pub content_hash: Vec<u8>,
    pub creation_source: String,
    pub source_autopilot_session_id: Option<Uuid>,
}

/// A dependency edge between two prompt template versions.
/// Used when one template includes or extends another.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredPromptTemplateDependency {
    pub prompt_template_version_id: Uuid,
    pub dependency_prompt_template_version_id: Uuid,
    pub dependency_key: String,
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unnecessary_wraps,
        reason = "`#[gtest]` tests return `googletest::Result<()>`"
    )]

    use googletest::prelude::*;
    use uuid::Uuid;

    use super::*;

    fn assert_round_trip<T>(value: T)
    where
        T: Clone + PartialEq + serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let serialized = serde_json::to_value(value.clone()).expect("serialize stored value");
        let round_tripped: T =
            serde_json::from_value(serialized).expect("deserialize stored value");
        expect_that!(&round_tripped, eq(&value));
    }

    #[gtest]
    fn prompt_template_round_trip() -> Result<()> {
        let source_body = "You are a helpful assistant.\n{{user_name}}".to_string();
        let template = StoredPromptTemplate {
            id: Uuid::now_v7(),
            template_key: "functions.my_func.variants.v1.system_template".to_string(),
            content_hash: vec![1, 2, 3, 4], // placeholder hash for test
            source_body,
            creation_source: "ui".to_string(),
            source_autopilot_session_id: None,
        };
        assert_round_trip(template);
        Ok(())
    }

    #[gtest]
    fn prompt_template_with_autopilot_session() -> Result<()> {
        let session_id = Uuid::now_v7();
        let template = StoredPromptTemplate {
            id: Uuid::now_v7(),
            template_key: "functions.my_func.variants.v1.user_template".to_string(),
            content_hash: vec![5, 6, 7, 8], // placeholder hash for test
            source_body: "Summarize: {{input}}".to_string(),
            creation_source: "autopilot".to_string(),
            source_autopilot_session_id: Some(session_id),
        };
        assert_round_trip(template);
        Ok(())
    }

    #[gtest]
    fn prompt_template_dependency_round_trip() -> Result<()> {
        let parent_template_id = Uuid::now_v7();
        let dependency = StoredPromptTemplateDependency {
            prompt_template_version_id: parent_template_id,
            dependency_prompt_template_version_id: Uuid::now_v7(),
            dependency_key: "shared.header".to_string(),
        };
        assert_round_trip(dependency);
        Ok(())
    }

    #[gtest]
    fn prompt_template_multiple_dependencies() -> Result<()> {
        let parent_id = Uuid::now_v7();
        let dep_edge1 = StoredPromptTemplateDependency {
            prompt_template_version_id: parent_id,
            dependency_prompt_template_version_id: Uuid::now_v7(),
            dependency_key: "shared.header".to_string(),
        };
        let dep_edge2 = StoredPromptTemplateDependency {
            prompt_template_version_id: parent_id,
            dependency_prompt_template_version_id: Uuid::now_v7(),
            dependency_key: "shared.footer".to_string(),
        };
        let dependencies = vec![dep_edge1, dep_edge2];
        assert_round_trip(dependencies);
        Ok(())
    }
}
