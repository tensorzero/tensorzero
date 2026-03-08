# minijinja-utils

Static analysis utilities for [MiniJinja](https://github.com/mitsuhiko/minijinja) templates.

## Overview

This crate analyzes MiniJinja templates to extract all template dependencies by parsing the template syntax and walking the AST. It identifies `include`, `import`, `from`, and `extends` statements and determines whether template loading can be statically resolved.

## Static vs Dynamic Template Loading

### Static Loads ✓

Template names that can be determined at parse time:

```jinja
{% include 'header.html' %}
{% extends 'base.html' %}
{% import 'macros.html' as m %}
{% include ['first.html', 'second.html'] %}
{% include 'a.html' if condition else 'b.html' %}
{% include 'optional.html' if condition %}
```

### Dynamic Loads ✗

Template names that depend on runtime values:

```jinja
{% include template_var %}
{% include get_template() %}
```

## API

### Main Function

```rust
pub fn collect_all_template_paths(
    env: &Environment<'_>,
    template_name: &str,
) -> Result<HashSet<PathBuf>, AnalysisError>
```

Recursively collects all template paths starting from a root template. Returns an error if any dynamic loads are found or parsing fails.

### Error Types

#### `AnalysisError`

```rust
pub enum AnalysisError {
    ParseError(minijinja::Error),
    DynamicLoadsFound(Vec<DynamicLoadLocation>),
}
```

- **`ParseError`**: Template has invalid MiniJinja syntax
- **`DynamicLoadsFound`**: One or more templates contain dynamic loads that cannot be statically resolved

#### `DynamicLoadLocation`

Provides detailed location information for dynamic loads:

- `template_name`: Which template contains the dynamic load
- `line`, `column`: 1-indexed position in the template
- `span`: Byte offsets of the expression
- `source_quote`: Extracted source showing the problem
- `reason`: Why it's dynamic (e.g., "variable", "conditional without else")
- `load_kind`: Type of statement (include, import, etc.)

#### `LoadKind`

```rust
pub enum LoadKind {
    Include { ignore_missing: bool },
    Import,
    FromImport,
    Extends,
}
```

Identifies which type of template loading statement was encountered.

## Example Usage

```rust
use minijinja::Environment;
use minijinja_utils::collect_all_template_paths;

let mut env = Environment::new();
env.add_template("main.html", "{% include 'header.html' %}Content").unwrap();
env.add_template("header.html", "Header").unwrap();

// Collect all template dependencies
let paths = collect_all_template_paths(&env, "main.html").unwrap();
assert_eq!(paths.len(), 2); // main.html and header.html
```

### Error Handling

```rust
use minijinja::Environment;
use minijinja_utils::{collect_all_template_paths, AnalysisError};

let mut env = Environment::new();
env.add_template("dynamic.html", "{% include template_var %}").unwrap();

match collect_all_template_paths(&env, "dynamic.html") {
    Err(AnalysisError::DynamicLoadsFound(locations)) => {
        for loc in locations {
            eprintln!("{}:{}:{}: {} - {}",
                loc.template_name, loc.line, loc.column,
                loc.load_kind, loc.reason);
        }
    }
    Err(AnalysisError::ParseError(e)) => {
        eprintln!("Parse error: {}", e);
    }
    Ok(paths) => {
        println!("Found {} templates", paths.len());
    }
}
```

## Supported MiniJinja Features

- **Include**: `{% include 'template.html' %}`
- **Import**: `{% import 'macros.html' as m %}`
- **From Import**: `{% from 'macros.html' import button %}`
- **Extends**: `{% extends 'base.html' %}`
- **List includes**: `{% include ['a.html', 'b.html'] %}`
- **Conditional with else**: `{% include 'a.html' if x else 'b.html' %}`
- **Circular dependencies**: Handled correctly without infinite loops

## Limitations

- **Requires static template names**: Cannot resolve variables, function calls, or expressions
- **Dependency on unstable API**: Uses MiniJinja's `unstable_machinery` feature which may change between versions

## Conditional Template Loading

When a template uses a conditional include without an else clause, the static template name is still extracted:

```jinja
{% include 'optional.html' if show_feature %}
```

This will:

- ✓ Extract `'optional.html'` as a dependency
- ✓ Require that `'optional.html'` exists at analysis time
- ⚠️ The template may or may not be loaded at runtime depending on the condition

This behavior enables static validation of all template paths while supporting conditional loading patterns. If the template name is dynamically computed (e.g., from a variable or function call), it will still be considered a dynamic load and produce an error.

## Implementation Note

This crate uses MiniJinja's `unstable_machinery` feature to access the AST. This API is marked unstable and may change between MiniJinja versions.
