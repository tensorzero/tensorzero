use crate::config::Namespace;
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::model::ModelTable;
use crate::variant::VariantConfig;

/// Validates that a variant's models (including models from candidate variants, recursively)
/// are compatible with the request namespace.
/// Returns an error if any model has a namespace that doesn't match the request namespace.
pub fn validate_variant_namespace_at_inference(
    variant_name: &str,
    variant_config: &VariantConfig,
    models: &ModelTable,
    request_namespace: Option<&Namespace>,
    function: &FunctionConfig,
) -> Result<(), Error> {
    for model_name in variant_config.all_model_names(function.variants()) {
        if let Some(model_namespace) = models.get_namespace(model_name)
            && request_namespace != Some(model_namespace)
        {
            return Err(ErrorDetails::InvalidRequest {
                message: format!(
                    "Variant `{variant_name}` uses model `{model_name}` which has namespace `{model_namespace}`, \
                    but the request namespace is {}. Namespaced models can only be used with a matching namespace.",
                    request_namespace.map_or_else(|| "`None`".to_string(), |ns| format!("`{ns}`"))
                ),
            }
            .into());
        }
    }
    Ok(())
}

/// Validates that a single model's namespace is compatible with the request namespace.
/// Returns an error if the model has a namespace that doesn't match the request namespace.
pub fn validate_model_namespace(
    model_name: &str,
    models: &ModelTable,
    request_namespace: Option<&Namespace>,
) -> Result<(), Error> {
    if let Some(model_namespace) = models.get_namespace(model_name)
        && request_namespace != Some(model_namespace)
    {
        return Err(ErrorDetails::InvalidRequest {
            message: format!(
                "Model `{model_name}` has namespace `{model_namespace}`, \
                but the request namespace is {}. Namespaced models can only be used with a matching namespace.",
                request_namespace.map_or_else(|| "`None`".to_string(), |ns| format!("`{ns}`"))
            ),
        }
        .into());
    }
    Ok(())
}
