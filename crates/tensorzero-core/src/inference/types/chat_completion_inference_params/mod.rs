pub use tensorzero_types::inference_params::{ChatCompletionInferenceParamsV2, ServiceTier};

pub fn warn_inference_parameter_not_supported(
    model_provider_name: &str,
    parameter_name: &str,
    suffix: Option<&str>,
) {
    let mut message = format!(
        "{model_provider_name} does not support the inference parameter `{parameter_name}`, so it will be ignored."
    );
    if let Some(suffix) = suffix {
        message.push_str(&format!(" {suffix}"));
    }
    tracing::warn!("{}", message);
}
