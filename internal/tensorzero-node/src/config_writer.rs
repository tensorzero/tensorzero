use config_writer::{ConfigWriter as InnerConfigWriter, EditPayload};
use tokio::sync::Mutex;

#[napi(js_name = "ConfigWriter")]
pub struct ConfigWriter {
    inner: Mutex<InnerConfigWriter>,
}

#[napi]
impl ConfigWriter {
    #[napi(factory)]
    pub async fn new(glob_pattern: String) -> Result<Self, napi::Error> {
        let inner = InnerConfigWriter::new(&glob_pattern)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to create ConfigWriter: {e}")))?;
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
