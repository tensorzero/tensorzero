use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use serde::Serialize;

use crate::{
    config::{path::ResolvedTomlPathData, ErrorContext, PathWithContents, SchemaData},
    error::{Error, ErrorDetails},
    inference::types::Role,
    jsonschema_util::StaticJSONSchema,
    variant::chat_completion::{
        TemplateWithSchema, UninitializedChatCompletionConfig, UninitializedInputWrappers,
    },
};

/// Holds of all of the templates and schemas used by a chat-completion variant.
#[derive(Debug, Default, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct ChatTemplates {
    #[serde(flatten)]
    templates: HashMap<String, Arc<TemplateWithSchema>>,
    #[serde(skip)]
    _private: (),
}

impl ChatTemplates {
    #[cfg(test)]
    pub fn empty() -> Self {
        Self {
            templates: HashMap::new(),
            _private: (),
        }
    }

    pub fn get_implicit_template(&self, role: Role) -> Option<&Arc<TemplateWithSchema>> {
        self.templates.get(role.implicit_template_name())
    }

    pub fn get_named_template(&self, name: &str) -> Option<&Arc<TemplateWithSchema>> {
        self.templates.get(name)
    }

    pub fn get_implicit_system_template(&self) -> Option<&Arc<TemplateWithSchema>> {
        self.templates.get("system")
    }

    pub fn get_all_explicit_template_names(&self) -> HashSet<String> {
        let mut names = HashSet::new();
        for (key, value) in &self.templates {
            // Exclude legacy templates with no schema - these templates
            // can only be invoked by a {`"type": "text", "text": "..."`} input block
            if !(value.legacy_definition && value.schema.is_none()) {
                names.insert(key.clone());
            }
        }
        names
    }

    pub fn get_all_template_paths(&self) -> Vec<&PathWithContents> {
        self.templates.values().map(|t| &t.template).collect()
    }

    /// Returns an iterator over all templates (name, template_with_schema pairs)
    pub(super) fn iter_templates(
        &self,
    ) -> impl Iterator<Item = (&String, &Arc<TemplateWithSchema>)> {
        self.templates.iter()
    }
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
        wrapper: Option<ResolvedTomlPathData>,
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
                legacy_definition: true,
            })),
            (None, None) => Ok(None),
            (Some(_), Some(_)) => Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "{error_prefix}: Cannot provide both `input_wrappers.{name}` and `{name}` template"
                ),
            })),
        }
    }

    /// Constructs a `ChatTemplates` from the templates defined in the `UninitializedChatCompletionConfig`,
    /// attaching the associated schemas from `SchemaData`.
    /// This handles both `input_wrappers` and `system_template`/`user_template`/`assistant_template` fields
    pub fn build(
        chat_config: &UninitializedChatCompletionConfig,
        schemas: &SchemaData,
        error_context: &ErrorContext,
    ) -> Result<Self, Error> {
        let function_and_variant_name = format!(
            "functions.{}.variants.{}",
            error_context.function_name, error_context.variant_name
        );
        let system = chat_config
            .system_template
            .as_ref()
            .map(|x| {
                Ok::<_, Error>(TemplateWithSchema {
                    template: PathWithContents::from_path(x.clone())?,
                    schema: schemas
                        .get_implicit_system_schema()
                        .map(|s| s.schema.clone()),
                    legacy_definition: true,
                })
            })
            .transpose()?;

        let user = chat_config
            .user_template
            .as_ref()
            .map(|x| {
                Ok::<_, Error>(TemplateWithSchema {
                    template: PathWithContents::from_path(x.clone())?,
                    schema: schemas.get_implicit_user_schema().map(|s| s.schema.clone()),
                    legacy_definition: true,
                })
            })
            .transpose()?;

        let assistant = chat_config
            .assistant_template
            .as_ref()
            .map(|x| {
                Ok::<_, Error>(TemplateWithSchema {
                    template: PathWithContents::from_path(x.clone())?,
                    schema: schemas
                        .get_implicit_assistant_schema()
                        .map(|s| s.schema.clone()),
                    legacy_definition: true,
                })
            })
            .transpose()?;

        let UninitializedInputWrappers {
            user: user_wrapper,
            assistant: assistant_wrapper,
            system: system_wrapper,
        } = chat_config.input_wrappers.clone().unwrap_or_default();

        let system = Self::validate_wrapper(
            system,
            schemas.get_implicit_system_schema().map(|s| &s.schema),
            system_wrapper,
            &function_and_variant_name,
            "system",
        )?;

        let user = Self::validate_wrapper(
            user,
            schemas.get_implicit_user_schema().map(|s| &s.schema),
            user_wrapper,
            &function_and_variant_name,
            "user",
        )?;

        let assistant = Self::validate_wrapper(
            assistant,
            schemas.get_implicit_assistant_schema().map(|s| &s.schema),
            assistant_wrapper,
            &function_and_variant_name,
            "assistant",
        )?;

        let mut templates = HashMap::new();
        if let Some(system) = system {
            templates.insert("system".to_string(), Arc::new(system));
        }
        if let Some(user) = user {
            templates.insert("user".to_string(), Arc::new(user));
        }
        if let Some(assistant) = assistant {
            templates.insert("assistant".to_string(), Arc::new(assistant));
        }

        for (template_name, template_config) in &chat_config.templates.inner {
            let template = TemplateWithSchema {
                template: PathWithContents::from_path(template_config.path.clone())?,
                schema: schemas
                    .get_named_schema(template_name)
                    .map(|s| s.schema.clone()),
                legacy_definition: false,
            };
            if templates
                .insert(template_name.clone(), Arc::new(template))
                .is_some()
            {
                // If we already have a template with the same name, then it must be `user_template`/`assistant_template`/`system_template`
                // (or a wrapper, but those are deprecated and undocumented)
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "{function_and_variant_name}: Cannot specify both `templates.{template_name}.path` and `{template_name}_template`"
                    ),
                }));
            }
        }

        Ok(ChatTemplates {
            templates,
            _private: (),
        })
    }
}
