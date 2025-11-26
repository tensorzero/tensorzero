use serde::Serialize;
use std::fmt;

/// Describes the type of template loading statement found during analysis.
///
/// MiniJinja supports four different ways to load or reference other templates.
/// This enum identifies which type of statement was encountered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum LoadKind {
    /// An `{% include %}` statement that embeds another template's content.
    ///
    /// The `ignore_missing` flag is `true` when using `{% include ... ignore missing %}`,
    /// which suppresses errors if the template doesn't exist.
    ///
    /// Example: `{% include 'header.html' %}`
    Include { ignore_missing: bool },

    /// An `{% import %}` statement that loads macros from another template.
    ///
    /// Example: `{% import 'macros.html' as m %}`
    Import,

    /// A `{% from ... import ... %}` statement that imports specific names from another template.
    ///
    /// Example: `{% from 'macros.html' import button, card %}`
    FromImport,

    /// An `{% extends %}` statement that specifies template inheritance.
    ///
    /// Example: `{% extends 'base.html' %}`
    Extends,
}

impl fmt::Display for LoadKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadKind::Include { ignore_missing } => {
                if *ignore_missing {
                    write!(f, "include (ignore missing)")
                } else {
                    write!(f, "include")
                }
            }
            LoadKind::Import => write!(f, "import"),
            LoadKind::FromImport => write!(f, "from import"),
            LoadKind::Extends => write!(f, "extends"),
        }
    }
}

/// Detailed location information for a dynamic template load.
///
/// This struct provides comprehensive debugging information when a template contains
/// a template loading expression that cannot be statically analyzed. It includes
/// the location, the problematic source code, and an explanation of why it's dynamic.
///
/// # Example Error Output
///
/// When formatted for display, a `DynamicLoadLocation` produces output like:
///
/// ```text
/// template.html:5:12: dynamic include - variable:
///   {% include template_name %}
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicLoadLocation {
    /// The name of the template containing the dynamic load.
    pub template_name: String,

    /// Line number (1-indexed) where the dynamic load statement begins.
    pub line: usize,

    /// Column number (1-indexed) where the dynamic expression starts.
    pub column: usize,

    /// Byte offsets (start, end) of the dynamic expression in the template source.
    ///
    /// These offsets can be used to extract or highlight the exact problematic code.
    pub span: (usize, usize),

    /// The extracted source text containing the dynamic expression.
    ///
    /// This typically shows the full template statement for context.
    pub source_quote: String,

    /// Human-readable explanation of why the load is considered dynamic.
    ///
    /// Common reasons include:
    /// - `"variable"`: Template name is a variable reference
    /// - `"function call"`: Template name comes from a function
    /// - `"filter"`: Template name uses a filter expression
    /// - `"non-string constant"`: Template name is not a string literal
    pub reason: String,

    /// The type of template loading statement that triggered the error.
    pub load_kind: LoadKind,
}

impl fmt::Display for DynamicLoadLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}: dynamic {} - {}:\n  {}",
            self.template_name,
            self.line,
            self.column,
            self.load_kind,
            self.reason,
            self.source_quote
        )
    }
}

/// Errors that can occur during template analysis.
#[derive(Debug, Serialize)]
#[serde(tag = "type", content = "details")]
pub enum AnalysisError {
    /// A template could not be parsed due to syntax errors.
    ///
    /// This wraps a MiniJinja parsing error. It occurs when a template has invalid
    /// MiniJinja syntax that prevents the parser from building an AST.
    ///
    /// # Example
    ///
    /// ```text
    /// Failed to parse template: unexpected end of template, expected end of block
    /// ```
    #[serde(serialize_with = "serialize_parse_error")]
    ParseError(minijinja::Error),

    /// One or more templates contain dynamic loads that cannot be statically resolved.
    ///
    /// This error indicates that static analysis is incomplete because some template
    /// loading expressions depend on runtime values. The vector contains detailed
    /// location information for each dynamic load found.
    ///
    /// # Example
    ///
    /// ```text
    /// Cannot statically analyze all template dependencies. Found 1 dynamic load(s):
    /// template.html:5:12: dynamic include - variable:
    ///   {% include template_name %}
    /// ```
    DynamicLoadsFound(Vec<DynamicLoadLocation>),
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnalysisError::ParseError(err) => {
                write!(f, "Failed to parse template: {err}")
            }
            AnalysisError::DynamicLoadsFound(locations) => {
                writeln!(
                    f,
                    "Cannot statically analyze all template dependencies. Found {} dynamic load(s):",
                    locations.len()
                )?;
                for (i, location) in locations.iter().enumerate() {
                    if i > 0 {
                        writeln!(f)?;
                    }
                    write!(f, "{location}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for AnalysisError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AnalysisError::ParseError(err) => Some(err),
            AnalysisError::DynamicLoadsFound(_) => None,
        }
    }
}

impl From<minijinja::Error> for AnalysisError {
    fn from(err: minijinja::Error) -> Self {
        AnalysisError::ParseError(err)
    }
}

impl PartialEq for AnalysisError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            // Compare ParseError by their string representation since minijinja::Error doesn't implement PartialEq
            (AnalysisError::ParseError(a), AnalysisError::ParseError(b)) => {
                a.to_string() == b.to_string()
            }
            (AnalysisError::DynamicLoadsFound(a), AnalysisError::DynamicLoadsFound(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for AnalysisError {}

/// Custom serialization for ParseError variant to convert minijinja::Error to a string
fn serialize_parse_error<S>(err: &minijinja::Error, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&err.to_string())
}
