use config_applier::{ConfigApplier as InnerConfigApplier, EditPayload};
use tokio::sync::Mutex;

#[napi(js_name = "ConfigApplier")]
pub struct ConfigApplier {
    inner: Mutex<InnerConfigApplier>,
}

#[napi]
impl ConfigApplier {
    #[napi(factory)]
    pub async fn new(glob_pattern: String) -> Result<Self, napi::Error> {
        let inner = InnerConfigApplier::new(&glob_pattern).await.map_err(|e| {
            napi::Error::from_reason(format!("Failed to create ConfigApplier: {e}"))
        })?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    #[napi]
    pub async fn apply_edit(&self, edit_json: String) -> Result<Vec<String>, napi::Error> {
        let edit: EditPayload = serde_json::from_str(&edit_json)
            .map_err(|e| napi::Error::from_reason(format!("Failed to parse EditPayload: {e}")))?;

        let paths = self
            .inner
            .lock()
            .await
            .apply_edit(&edit)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to apply edit: {e}")))?;

        Ok(paths.into_iter().map(|p| p.display().to_string()).collect())
    }
}
