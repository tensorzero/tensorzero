use serde::Serialize;

use crate::{
    config::{path::ResolvedTomlPath, PathWithContents, SchemaData},
    error::{Error, ErrorDetails},
    jsonschema_util::StaticJSONSchema,
    variant::chat_completion::{TemplateWithSchema, UninitializedInputWrappers},
};

/// Holds of all of the templates and schemas used by a chat-completion variant.
#[derive(Debug, Default, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct ChatTemplates {
    pub system: Option<TemplateWithSchema>,
    pub user: Option<TemplateWithSchema>,
    pub assistant: Option<TemplateWithSchema>,
}

impl ChatTemplates {
    // Checks that an `input_wrappers` field is not used at the same time as
    // a `user_template`/`assistant_template`/`system_template` field or
    // schema.
    // Returns the final `TemplateWithSchema` to use (assuming we have a template).
    // After this point, we no longer track whether a template came from `input_wrappers` -
    // the runtime behavior is determined by whether or not `TemplateWithSchema.schema`
    // is set.
    fn validate_wrapper(
        template_and_schema: Option<TemplateWithSchema>,
        schema: Option<&StaticJSONSchema>,
        wrapper: Option<ResolvedTomlPath>,
        error_prefix: &str,
        name: &str,
    ) -> Result<Option<TemplateWithSchema>, Error> {
        // If both a function schema and an input wrapper are provided, error,
        // as input wrappers just take in a plain text input.
        if schema.is_some() && wrapper.is_some() {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "{error_prefix}: Cannot provide both `input_wrappers.{name}` and `{name}_schema`"
                ),
            }));
        }
        // Check the merged 'TemplateWithSchema' (the function template combined with a non-input-wrapper template)
        // We don't allow specifying both a normal template and an input wrapper template.
        match (template_and_schema, wrapper) {
            // We have a `user`/`assistant`/`system` template and no corresponding `input_wrappers`,
            // entry, so use our existing
            (Some(schema), None) => Ok(Some(schema)),
            // If we just have an input wrapper, then we create a new 'TemplateWithSchema'
            // with no schema,
            (None, Some(wrapper)) => Ok(Some(TemplateWithSchema {
                template: PathWithContents::from_path(wrapper)?,
                schema: None,
            })),
            (None, None) => Ok(None),
            (Some(_), Some(_)) => Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "{error_prefix}: Cannot provide both `input_wrappers.{name}` and `{name}` template"
                ),
            })),
        }
    }
    /// Applies the templates from `input_wrappers`,
    /// erroring if we already have a template specified
    pub fn apply_wrappers(
        self,
        input_wrappers: Option<UninitializedInputWrappers>,
        schemas: &SchemaData,
        // A string like 'functions.<function_name>.variants.<variant_name>', used in error messages
        function_and_variant_name: &str,
    ) -> Result<Self, Error> {
        let UninitializedInputWrappers {
            user: user_wrapper,
            assistant: assistant_wrapper,
            system: system_wrapper,
        } = input_wrappers.unwrap_or_default();
        Ok(ChatTemplates {
            system: Self::validate_wrapper(
                self.system,
                schemas.system.as_ref(),
                system_wrapper,
                function_and_variant_name,
                "system",
            )?,
            user: Self::validate_wrapper(
                self.user,
                schemas.user.as_ref(),
                user_wrapper,
                function_and_variant_name,
                "user",
            )?,
            assistant: Self::validate_wrapper(
                self.assistant,
                schemas.assistant.as_ref(),
                assistant_wrapper,
                function_and_variant_name,
                "assistant",
            )?,
        })
    }
}
