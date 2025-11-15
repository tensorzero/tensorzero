use std::fmt;

/// Describes the template loading statement that triggered analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadKind {
    Include { ignore_missing: bool },
    Import,
    FromImport,
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

/// Information about a dynamic template load that prevents complete static analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicLoadLocation {
    /// The name of the template containing the dynamic load.
    pub template_name: String,
    /// Line number (1-indexed) where the dynamic load occurs.
    pub line: usize,
    /// Column number (1-indexed) where the dynamic load occurs.
    pub column: usize,
    /// Byte offsets (start, end) of the expression in the template source.
    pub span: (usize, usize),
    /// Extracted source text showing the dynamic expression.
    pub source_quote: String,
    /// Explanation of why the load is dynamic (e.g., "variable", "conditional without else").
    pub reason: String,
    /// The type of template loading statement (include, import, etc.).
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
#[derive(Debug)]
pub enum AnalysisError {
    /// Failed to parse a template.
    ParseError(minijinja::Error),
    /// Found one or more dynamic template loads that cannot be statically analyzed.
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
