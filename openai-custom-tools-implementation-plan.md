# OpenAI Custom Tools Implementation Plan

## Executive Summary

This document outlines the implementation plan for adding support for OpenAI's custom tools (with text/grammar formats) to TensorZero. The implementation will maintain full backward compatibility with the existing `ClientSideFunctionTool` format while adding support for OpenAI's newer custom tool format.

### Goals
- Support OpenAI's custom tools that can be passed dynamically
- Maintain backward compatibility with existing `additional_tools` field
- Use wrapper type with custom deserializer (try `Tool` first, fallback to `ClientSideFunctionTool`)
- Normalize to internal format for storage
- **Add new iterator that includes custom tools; existing iterators filter them out**
- **Only OpenAI provider uses the new iterator with custom tools**
- Leverage existing tool choice/allowed tools patterns
- Generate proper Python/TypeScript types

### Key Architecture Decision: Iterator Strategy

**Custom tools will be filtered by default in existing iterators:**
- `tools_available()` → will filter out custom tools (backward compatible)
- `strict_tools_available()` → will filter out custom tools
- `get_scoped_provider_tools()` → unchanged (provider tools)

**New iterator for OpenAI:**
- `all_tools_including_custom()` → returns ALL tools including custom
- Only OpenAI provider will call this new method
- All other providers continue using existing iterators (automatically filtered)

This approach:
- ✅ Maintains backward compatibility (existing code sees no custom tools)
- ✅ No need to add filtering logic to each provider
- ✅ Explicit opt-in for custom tools (only OpenAI)
- ✅ Clear API separation

---

## Background: Current Tool Architecture

### Current Tool Type Hierarchy

**Location:** `tensorzero-core/src/tool.rs`

```rust
// Lines 66-68
pub enum Tool {
    ClientSideFunction(ClientSideFunctionTool),
}

// Lines 155-166
pub struct ClientSideFunctionTool {
    pub description: String,
    pub parameters: Value,  // JSON Schema
    pub name: String,
    pub strict: bool,
}
```

**Key Comment (lines 50-61):**
> `Tool` is the generic form for all tools that TensorZero itself manages. Today, this is only ClientSideFunctionTools (the original kind), but soon we'll implement OpenAI's custom tools standard, MCP, and potentially more. Most likely, this will eventually become the wire type too with a custom deserializer so that folks can specify ClientSideFunctionTools without tags but then can add tags and specify other kinds of tool.

### ToolCallConfig Structure

**Location:** `tensorzero-core/src/tool.rs:347-356`

```rust
pub struct ToolCallConfig {
    pub(crate) static_tools_available: Vec<ToolConfig>,
    pub(crate) dynamic_tools_available: Vec<ToolConfig>,
    pub provider_tools: Vec<ProviderTool>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: Option<bool>,
    pub allowed_tools: AllowedTools,
}
```

**Key Methods (to be updated):**
- `tools_available()` - Returns ALL tools (static + dynamic) → **WILL FILTER CUSTOM**
- `strict_tools_available()` - Returns tools filtered by `allowed_tools` list → **WILL FILTER CUSTOM**
- `get_scoped_provider_tools()` - Returns provider-specific tools → **UNCHANGED**
- **NEW:** `all_tools_including_custom()` - Returns everything including custom tools

### DynamicToolParams (Wire Format)

**Location:** `tensorzero-core/src/tool.rs:1255-1268`

```rust
pub struct DynamicToolParams {
    pub allowed_tools: Option<Vec<String>>,
    pub additional_tools: Option<Vec<ClientSideFunctionTool>>,  // ← Will change to Vec<DynamicTool>
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub provider_tools: Vec<ProviderTool>,
}
```

### Database Storage

**Location:** `tensorzero-core/src/tool.rs:650-660`

```rust
pub struct ToolCallConfigDatabaseInsert {
    pub dynamic_tools: Vec<Tool>,  // ← Already stores Tool enum
    pub dynamic_provider_tools: Vec<ProviderTool>,
    pub allowed_tools: AllowedTools,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: Option<bool>,
    tool_params: LegacyToolCallConfigDatabaseInsert,
}
```

### OpenAI Compatible Interface

**Location:** `tensorzero-core/src/endpoints/openai_compatible.rs:387-395`

```rust
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", content = "function")]
#[serde(rename_all = "snake_case")]
enum OpenAICompatibleTool {
    Function {
        description: Option<String>,
        name: String,
        parameters: Value,
        #[serde(default)]
        strict: bool,
    },
}
```

**Conversion (lines 1177-1192):**
```rust
impl From<OpenAICompatibleTool> for ClientSideFunctionTool {
    fn from(tool: OpenAICompatibleTool) -> Self {
        match tool {
            OpenAICompatibleTool::Function { description, name, parameters, strict } => {
                ClientSideFunctionTool {
                    description: description.unwrap_or_default(),
                    parameters,
                    name,
                    strict,
                }
            }
        }
    }
}
```

**Integration into DynamicToolParams (lines 776-781):**
```rust
let dynamic_tool_params = DynamicToolParams {
    allowed_tools,
    additional_tools: openai_compatible_params
        .tools
        .map(|tools| tools.into_iter().map(OpenAICompatibleTool::into).collect()),
    tool_choice,
    parallel_tool_calls: openai_compatible_params.parallel_tool_calls,
    provider_tools: openai_compatible_params.tensorzero_provider_tools,
};
```

### Tool Processing Pipeline

1. **API Request** → `OpenAICompatibleParams` or `ClientInferenceParams`
   - OpenAI endpoint: `tools` field converted to `additional_tools`
   - Native endpoint: Direct `additional_tools` field

2. **Inference Handler** (`inference()` in `endpoints/inference.rs:229`)
   - Calls `function.prepare_tool_config(params.dynamic_tool_params, &config.tools)`
   - Creates `ToolCallConfig` with merged static + dynamic tools

3. **Provider Conversion** (e.g., OpenAI in `providers/openai/mod.rs:1520`)
   - Calls `tool_config.tools_available()` to iterate ALL tools
   - Converts each tool via `Into::into` to provider format

### OpenAI Provider Tool Handling

**Location:** `tensorzero-core/src/providers/openai/mod.rs:2000-2012`

```rust
impl<'a> From<&'a ToolConfig> for OpenAITool<'a> {
    fn from(tool: &'a ToolConfig) -> Self {
        OpenAITool {
            r#type: OpenAIToolType::Function,
            function: OpenAIFunction {
                name: tool.name(),
                description: Some(tool.description()),
                parameters: tool.parameters(),
            },
            strict: tool.strict(),
        }
    }
}
```

**Allowed Tools Handling (lines 1531-1554):**
When `allowed_tools` is set, creates OpenAI `AllowedToolsChoice` structure with list of allowed tool names.

### Current Provider Iteration Patterns

**Providers using `tools_available()` (all tools):**
- OpenAI (`providers/openai/mod.rs:1527`)
- Groq (`providers/groq.rs:586`)
- Mistral (`providers/mistral.rs:480`)
- Gemini variants (various)

**Providers using `strict_tools_available()` (respects allowed_tools):**
- Anthropic (`providers/anthropic.rs:795`)
- AWS Bedrock (`providers/aws_bedrock.rs:255, 400`)

---

## OpenAI Custom Tools API Reference

From OpenAI documentation, tools can be either "function" or "custom" type:

### Function Tool
```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get weather for a location",
    "parameters": { /* JSON Schema */ }
  }
}
```

### Custom Tool
```json
{
  "type": "custom",
  "custom": {
    "name": "my_custom_tool",
    "description": "Optional description",
    "format": {
      "type": "text"  // or "grammar"
    }
  }
}
```

### Grammar Format
```json
{
  "type": "custom",
  "custom": {
    "name": "structured_parser",
    "format": {
      "type": "grammar",
      "grammar": {
        "syntax": "lark",  // or "regex"
        "definition": "/* grammar definition */"
      }
    }
  }
}
```

### Tool Choice with Allowed Tools
```json
{
  "tool_choice": {
    "type": "allowed_tools",
    "allowed_tools": {
      "mode": "auto",  // or "required"
      "tools": [
        { "type": "function", "function": { "name": "get_weather" } },
        { "type": "custom", "custom": { "name": "my_tool" } }
      ]
    }
  }
}
```

---

## Implementation Plan

## Phase 1: Core Type System

### File: `tensorzero-core/src/tool.rs`

#### 1.1 Add Custom Tool Structures

Add after `ClientSideFunctionTool` definition (around line 166):

```rust
/// `CustomTool` represents OpenAI's custom tool format, which allows
/// for text or grammar-based tool definitions beyond standard function calling.
#[derive(ts_rs::TS, Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[ts(export)]
#[serde(deny_unknown_fields)]
#[export_schema]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct CustomTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<CustomToolFormat>,
}

#[derive(ts_rs::TS, Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CustomToolFormat {
    Text,
    Grammar {
        grammar: GrammarDefinition,
    },
}

#[derive(ts_rs::TS, Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[ts(export)]
pub struct GrammarDefinition {
    pub syntax: GrammarSyntax,
    pub definition: String,
}

#[derive(ts_rs::TS, Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum GrammarSyntax {
    Lark,
    Regex,
}
```

Add PyO3 methods for CustomTool:

```rust
#[cfg(feature = "pyo3")]
#[pymethods]
impl CustomTool {
    #[getter]
    pub fn get_name(&self) -> &str {
        &self.name
    }

    #[getter]
    pub fn get_description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    #[getter]
    pub fn get_format<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.format {
            Some(format) => {
                serialize_to_dict(py, format.clone()).map(|x| Some(x.into_bound(py)))
            }
            None => Ok(None),
        }
    }

    pub fn __repr__(&self) -> String {
        format!("CustomTool(name='{}')", self.name)
    }
}
```

#### 1.2 Extend Tool Enum

Modify the `Tool` enum (line 66):

```rust
#[derive(ts_rs::TS, AsRefStr, Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub enum Tool {
    ClientSideFunction(ClientSideFunctionTool),
    Custom(CustomTool),  // ← NEW
}
```

#### 1.3 Update Tool Methods

Update `Tool::name()` (line 78):

```rust
fn name(&self) -> &str {
    match self {
        Tool::ClientSideFunction(tool) => &tool.name,
        Tool::Custom(tool) => &tool.name,  // ← NEW
    }
}
```

Update `Tool::into_dynamic_tool_config()` (line 84):

**Note:** This method is only used for function tools that need parameter validation. Custom tools should not go through this path. We need to verify all call sites.

```rust
fn into_dynamic_tool_config(self) -> DynamicToolConfig {
    match self {
        Tool::ClientSideFunction(tool) => DynamicToolConfig {
            description: tool.description,
            parameters: DynamicJSONSchema::new(tool.parameters),
            name: tool.name,
            strict: tool.strict,
        },
        Tool::Custom(_) => {
            // Custom tools don't have JSON schema parameters and shouldn't
            // go through parameter validation. This path should not be reached
            // in normal operation.
            unreachable!("Custom tools do not support parameter validation")
        }
    }
}
```

**TODO:** Audit all call sites of `into_dynamic_tool_config()` to ensure custom tools are filtered before reaching this method.

Update PyO3 methods (lines 98-143) to handle custom tools:

```rust
#[cfg(feature = "pyo3")]
#[pymethods]
impl Tool {
    #[getter]
    pub fn get_type(&self) -> &str {
        self.as_ref()
    }

    #[getter]
    pub fn get_parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Tool::ClientSideFunction(tool) => {
                serialize_to_dict(py, tool.parameters.clone()).map(|x| x.into_bound(py))
            }
            Tool::Custom(_) => {
                Err(pyo3::exceptions::PyAttributeError::new_err(
                    "Custom tools do not have parameters. Check type field first."
                ))
            }
        }
    }

    #[getter]
    pub fn get_description(&self) -> PyResult<String> {
        match self {
            Tool::ClientSideFunction(tool) => Ok(tool.description.clone()),
            Tool::Custom(tool) => {
                tool.description.clone().ok_or_else(|| {
                    pyo3::exceptions::PyAttributeError::new_err(
                        "This custom tool has no description"
                    )
                })
            }
        }
    }

    #[getter]
    pub fn get_name(&self) -> &str {
        match self {
            Tool::ClientSideFunction(tool) => &tool.name,
            Tool::Custom(tool) => &tool.name,
        }
    }

    #[getter]
    pub fn get_strict(&self) -> PyResult<bool> {
        match self {
            Tool::ClientSideFunction(tool) => Ok(tool.strict),
            Tool::Custom(_) => {
                Err(pyo3::exceptions::PyAttributeError::new_err(
                    "Custom tools do not have strict mode. Check type field first."
                ))
            }
        }
    }

    #[getter]
    pub fn get_format<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Tool::Custom(tool) => {
                match &tool.format {
                    Some(format) => {
                        serialize_to_dict(py, format.clone()).map(|x| x.into_bound(py))
                    }
                    None => Ok(py.None().into_bound(py)),
                }
            }
            Tool::ClientSideFunction(_) => {
                Err(pyo3::exceptions::PyAttributeError::new_err(
                    "Function tools do not have format. Check type field first."
                ))
            }
        }
    }

    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}
```

#### 1.4 Create DynamicTool Wrapper with Custom Deserializer

Add new type after `Tool` definition (around line 200):

```rust
/// `DynamicTool` is a wrapper around `Tool` that provides backward compatibility
/// with the legacy untagged `ClientSideFunctionTool` format. It uses a custom
/// deserializer that first tries to parse as a tagged `Tool` enum, and falls back
/// to parsing as an untagged `ClientSideFunctionTool` (wrapping it as
/// `Tool::ClientSideFunction`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DynamicTool(pub Tool);

impl<'de> Deserialize<'de> for DynamicTool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;

        // First, try to deserialize as a tagged Tool (new format)
        if let Ok(tool) = serde_json::from_value::<Tool>(value.clone()) {
            return Ok(DynamicTool(tool));
        }

        // Fall back to legacy untagged ClientSideFunctionTool format
        match serde_json::from_value::<ClientSideFunctionTool>(value) {
            Ok(function_tool) => Ok(DynamicTool(Tool::ClientSideFunction(function_tool))),
            Err(e) => Err(serde::de::Error::custom(format!(
                "Failed to parse as Tool or ClientSideFunctionTool: {}",
                e
            ))),
        }
    }
}

impl From<DynamicTool> for Tool {
    fn from(dynamic_tool: DynamicTool) -> Self {
        dynamic_tool.0
    }
}

impl From<Tool> for DynamicTool {
    fn from(tool: Tool) -> Self {
        DynamicTool(tool)
    }
}
```

#### 1.5 Update DynamicToolParams

Modify `DynamicToolParams` (line 1255):

```rust
#[derive(ts_rs::TS, Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[ts(export)]
#[export_schema]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct DynamicToolParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub additional_tools: Option<Vec<DynamicTool>>,  // ← CHANGED from Vec<ClientSideFunctionTool>

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    #[serde(default)]
    pub provider_tools: Vec<ProviderTool>,
}
```

**Impact:** This change will require updates in several places where `additional_tools` is accessed:
- `tensorzero-core/src/function.rs:342` - `prepare_tool_config()`
- `tensorzero-core/src/endpoints/openai_compatible.rs:776-781` - conversion

#### 1.6 Add Helper Method to Check if Tool is Custom

Add to `Tool` impl block:

```rust
impl Tool {
    // ... existing methods ...

    /// Returns true if this is a custom tool (not a function tool)
    pub fn is_custom(&self) -> bool {
        matches!(self, Tool::Custom(_))
    }

    /// Returns true if this is a function tool
    pub fn is_function(&self) -> bool {
        matches!(self, Tool::ClientSideFunction(_))
    }
}
```

Add similar methods to `ToolConfig`:

```rust
impl ToolConfig {
    // ... existing methods ...

    /// Returns true if this is a custom tool (not a function tool)
    pub fn is_custom(&self) -> bool {
        match self {
            ToolConfig::Static(_) => false,  // Static tools are always functions
            ToolConfig::Dynamic(dynamic) => dynamic.tool.is_custom(),
        }
    }

    /// Returns true if this is a function tool
    pub fn is_function(&self) -> bool {
        !self.is_custom()
    }
}
```

---

## Phase 2: Update ToolCallConfig Iterators

### File: `tensorzero-core/src/tool.rs`

**Critical Change:** Modify existing iterators to filter out custom tools, and add new iterator for all tools.

#### 2.1 Update `tools_available()` to Filter Custom Tools

Find `tools_available()` method (around line 400-410):

**BEFORE:**
```rust
pub fn tools_available(&self) -> impl Iterator<Item = &ToolConfig> {
    self.static_tools_available
        .iter()
        .chain(self.dynamic_tools_available.iter())
}
```

**AFTER:**
```rust
/// Returns all function tools available (filters out custom tools).
/// This is the default iterator that maintains backward compatibility.
/// For custom tools, use `all_tools_including_custom()`.
pub fn tools_available(&self) -> impl Iterator<Item = &ToolConfig> {
    self.static_tools_available
        .iter()
        .chain(self.dynamic_tools_available.iter())
        .filter(|tool| tool.is_function())  // ← NEW: Filter custom tools
}
```

#### 2.2 Update `strict_tools_available()` to Filter Custom Tools

Find `strict_tools_available()` method:

**Add filtering for custom tools:**

```rust
/// Returns tools filtered by allowed_tools list (and also filters out custom tools).
/// For custom tools, use `all_tools_including_custom()` with manual filtering.
pub fn strict_tools_available(&self) -> impl Iterator<Item = &ToolConfig> {
    // ... existing allowed_tools filtering logic ...
    // Add .filter(|tool| tool.is_function()) to the final iterator
}
```

#### 2.3 Add New Iterator for All Tools Including Custom

Add new method to `ToolCallConfig`:

```rust
/// Returns ALL tools including custom tools.
/// **Only use this for providers that explicitly support custom tools (currently only OpenAI).**
/// Most providers should use `tools_available()` which filters out custom tools.
pub fn all_tools_including_custom(&self) -> impl Iterator<Item = &ToolConfig> {
    self.static_tools_available
        .iter()
        .chain(self.dynamic_tools_available.iter())
    // No filtering - returns everything
}
```

#### 2.4 Add Filtered Version for OpenAI with Allowed Tools

Add method for OpenAI to get all tools respecting allowed_tools constraint:

```rust
/// Returns tools filtered by allowed_tools list, including custom tools.
/// This is specifically for OpenAI which supports both function and custom tools.
pub fn strict_tools_including_custom(&self) -> impl Iterator<Item = &ToolConfig> {
    let tool_names: Option<HashSet<&str>> = match &self.allowed_tools {
        AllowedTools::FunctionDefault => None,  // Use all function tools (no custom)
        AllowedTools::AllAllowedTools(names) | AllowedTools::DynamicAllowedTools(names) => {
            Some(names.iter().map(|s| s.as_str()).collect())
        }
    };

    self.all_tools_including_custom().filter(move |tool| {
        if let Some(ref allowed) = tool_names {
            allowed.contains(tool.name())
        } else {
            // FunctionDefault mode - only include function tools
            tool.is_function()
        }
    })
}
```

**Key Behavior:**
- `FunctionDefault` → Only function tools (backward compatible)
- `AllAllowedTools` or `DynamicAllowedTools` → Respects tool name list, includes custom if specified

---

## Phase 3: OpenAI Compatible Interface

### File: `tensorzero-core/src/endpoints/openai_compatible.rs`

#### 3.1 Extend OpenAICompatibleTool Enum

Modify `OpenAICompatibleTool` (line 387):

**Current structure issue:** The current enum uses `#[serde(tag = "type", content = "function")]` which means it expects:
```json
{"type": "function", "function": {...}}
```

But we need to support:
```json
{"type": "function", "function": {...}}
{"type": "custom", "custom": {...}}
```

**Solution:** Use externally tagged enum without content attribute:

```rust
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAICompatibleTool {
    Function {
        function: OpenAIFunctionDefinition,
    },
    Custom {
        custom: OpenAICustomDefinition,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct OpenAIFunctionDefinition {
    description: Option<String>,
    name: String,
    parameters: Value,
    #[serde(default)]
    strict: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct OpenAICustomDefinition {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<OpenAICustomFormat>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAICustomFormat {
    Text,
    Grammar {
        grammar: OpenAIGrammarDefinition,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct OpenAIGrammarDefinition {
    syntax: String,  // "lark" or "regex"
    definition: String,
}
```

#### 3.2 Update Conversion to Tool

Replace the `From<OpenAICompatibleTool> for ClientSideFunctionTool` impl (line 1177) with:

```rust
impl From<OpenAICompatibleTool> for Tool {
    fn from(tool: OpenAICompatibleTool) -> Self {
        match tool {
            OpenAICompatibleTool::Function { function } => {
                Tool::ClientSideFunction(ClientSideFunctionTool {
                    description: function.description.unwrap_or_default(),
                    parameters: function.parameters,
                    name: function.name,
                    strict: function.strict,
                })
            }
            OpenAICompatibleTool::Custom { custom } => {
                Tool::Custom(CustomTool {
                    name: custom.name,
                    description: custom.description,
                    format: custom.format.map(|f| match f {
                        OpenAICustomFormat::Text => CustomToolFormat::Text,
                        OpenAICustomFormat::Grammar { grammar } => {
                            CustomToolFormat::Grammar {
                                grammar: GrammarDefinition {
                                    syntax: match grammar.syntax.as_str() {
                                        "lark" => GrammarSyntax::Lark,
                                        "regex" => GrammarSyntax::Regex,
                                        _ => GrammarSyntax::Lark, // Default fallback
                                    },
                                    definition: grammar.definition,
                                },
                            }
                        }
                    }),
                })
            }
        }
    }
}
```

#### 3.3 Update Tool Conversion Call

Modify the conversion in `inference()` (lines 776-781):

```rust
let dynamic_tool_params = DynamicToolParams {
    allowed_tools,
    additional_tools: openai_compatible_params
        .tools
        .map(|tools| tools.into_iter()
            .map(|t| DynamicTool(t.into()))  // ← Convert to Tool then wrap in DynamicTool
            .collect()
        ),
    tool_choice,
    parallel_tool_calls: openai_compatible_params.parallel_tool_calls,
    provider_tools: openai_compatible_params.tensorzero_provider_tools,
};
```

---

## Phase 4: Update Tool Processing

### File: `tensorzero-core/src/function.rs`

#### 4.1 Update prepare_tool_config()

Modify `prepare_tool_config()` (around line 342) to handle `DynamicTool`:

**Key changes:**
- Update type from `Vec<ClientSideFunctionTool>` to `Vec<DynamicTool>`
- Extract inner `Tool` from `DynamicTool` wrapper
- Handle both function and custom tools

```rust
pub fn prepare_tool_config(
    &self,
    dynamic_tool_params: Option<DynamicToolParams>,
    tools: &HashMap<String, ToolConfig>,
) -> Result<ToolCallConfig, Error> {
    // ... existing code for static tools ...

    // Process dynamic tools
    let dynamic_tools = dynamic_tool_params
        .as_ref()
        .and_then(|params| params.additional_tools.as_ref())
        .map(|tools| {
            tools
                .iter()
                .map(|dynamic_tool| {
                    // Extract the Tool from DynamicTool wrapper
                    let tool: &Tool = &dynamic_tool.0;

                    // Convert to DynamicToolConfig
                    // Note: Custom tools will be stored but won't be validated
                    // They'll be filtered by default iterators
                    match tool {
                        Tool::ClientSideFunction(_) => {
                            // Existing logic for function tools
                            // ... validation, schema compilation, etc ...
                        }
                        Tool::Custom(_) => {
                            // Custom tools don't need parameter validation
                            // Just store them as-is
                            Ok(ToolConfig::Dynamic(/* ... */))
                        }
                    }
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    // ... rest of function ...
    // Existing duplicate checking should work with custom tools too
}
```

**Note:** Need to examine the actual implementation to see how `DynamicToolConfig` is constructed. Custom tools may need special handling.

---

## Phase 5: Provider Integration

### 5.1 OpenAI Provider - Use New Iterator

**File:** `tensorzero-core/src/providers/openai/mod.rs`

#### Update Tool Preparation Function

Find `prepare_openai_tools()` (around line 1520):

**BEFORE:**
```rust
let tools = Some(tool_config.tools_available().map(Into::into).collect());
```

**AFTER:**
```rust
// Use the new iterator that includes custom tools
let tools = Some(
    tool_config
        .all_tools_including_custom()  // ← NEW: Get all tools including custom
        .map(Into::into)
        .collect()
);
```

Or if respecting allowed_tools:

```rust
let tools = Some(
    tool_config
        .strict_tools_including_custom()  // ← NEW: Respect allowed_tools but include custom
        .map(Into::into)
        .collect()
);
```

#### Add Custom Tool Output Structures

Add after existing `OpenAITool` definition (around line 2000):

```rust
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAIToolOutput<'a> {
    Function {
        function: OpenAIFunction<'a>,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    Custom {
        custom: OpenAICustomToolOutput<'a>,
    },
}

#[derive(Debug, Serialize)]
struct OpenAICustomToolOutput<'a> {
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<OpenAICustomFormatOutput<'a>>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAICustomFormatOutput<'a> {
    Text,
    Grammar {
        grammar: OpenAIGrammarOutput<'a>,
    },
}

#[derive(Debug, Serialize)]
struct OpenAIGrammarOutput<'a> {
    syntax: &'a str,
    definition: &'a str,
}
```

#### Update Tool Conversion

Modify or add new conversion logic:

```rust
impl<'a> From<&'a ToolConfig> for OpenAIToolOutput<'a> {
    fn from(tool: &'a ToolConfig) -> Self {
        match tool {
            ToolConfig::Static(static_tool) => {
                OpenAIToolOutput::Function {
                    function: OpenAIFunction {
                        name: &static_tool.name,
                        description: Some(&static_tool.description),
                        parameters: &static_tool.parameters,
                    },
                    strict: if static_tool.strict { Some(true) } else { None },
                }
            }
            ToolConfig::Dynamic(dynamic_tool) => {
                match &dynamic_tool.tool {
                    Tool::ClientSideFunction(func) => {
                        OpenAIToolOutput::Function {
                            function: OpenAIFunction {
                                name: &func.name,
                                description: Some(&func.description),
                                parameters: &func.parameters,
                            },
                            strict: if func.strict { Some(true) } else { None },
                        }
                    }
                    Tool::Custom(custom) => {
                        OpenAIToolOutput::Custom {
                            custom: OpenAICustomToolOutput {
                                name: &custom.name,
                                description: custom.description.as_deref(),
                                format: custom.format.as_ref().map(|f| match f {
                                    CustomToolFormat::Text => OpenAICustomFormatOutput::Text,
                                    CustomToolFormat::Grammar { grammar } => {
                                        OpenAICustomFormatOutput::Grammar {
                                            grammar: OpenAIGrammarOutput {
                                                syntax: match grammar.syntax {
                                                    GrammarSyntax::Lark => "lark",
                                                    GrammarSyntax::Regex => "regex",
                                                },
                                                definition: &grammar.definition,
                                            },
                                        }
                                    }
                                }),
                            },
                        }
                    }
                }
            }
        }
    }
}
```

### 5.2 Other Providers - No Changes Needed!

**Key Insight:** Because we modified `tools_available()` and `strict_tools_available()` to filter out custom tools, all other providers automatically ignore custom tools without any code changes!

**Providers that automatically filter custom tools:**
- Anthropic (`providers/anthropic.rs`) - uses `strict_tools_available()`
- AWS Bedrock (`providers/aws_bedrock.rs`) - uses `strict_tools_available()`
- Groq (`providers/groq.rs`) - uses `tools_available()`
- Mistral (`providers/mistral.rs`) - uses `tools_available()`
- Together (`providers/together.rs`) - uses `tools_available()`
- Fireworks (`providers/fireworks/mod.rs`) - uses `tools_available()`
- Azure (`providers/azure.rs`) - uses `tools_available()`
- GCP Vertex Gemini (`providers/gcp_vertex_gemini/mod.rs`) - uses `tools_available()`
- DeepSeek (`providers/deepseek.rs`) - uses `tools_available()`

**No changes needed!** This is the beauty of the iterator strategy.

---

## Phase 6: Storage & Database

### File: `tensorzero-core/src/tool.rs`

The `ToolCallConfigDatabaseInsert` (lines 650-660) already stores `Vec<Tool>`, so custom tools should serialize automatically.

**Verification needed:**
1. Test that `Tool::Custom` serializes/deserializes correctly from JSON
2. Test that database round-trip preserves custom tools
3. Verify ClickHouse Array(String) column handles serialized custom tools

**No code changes expected** - just need tests to verify.

---

## Phase 7: Python & TypeScript Bindings

### 7.1 Export Rust Types

Ensure all new types have proper derives:
- `#[derive(ts_rs::TS)]` with `#[ts(export)]` ✅
- `#[export_schema]` for JSON schema generation ✅
- `#[cfg_attr(feature = "pyo3", pyclass)]` for Python bindings ✅

### 7.2 Generate Bindings

```bash
# Generate TypeScript bindings
cd internal/tensorzero-node
pnpm build-bindings

# Generate Python bindings (schemas)
cd tensorzero-core
cargo test export_bindings
```

### 7.3 Python Type Hints

**File:** `clients/python/tensorzero/tensorzero.pyi`

Verify generated types appear automatically. Expected:

```python
class CustomTool:
    name: str
    description: Optional[str]
    format: Optional[CustomToolFormat]

class CustomToolFormat:
    type: Literal["text", "grammar"]
    # ... grammar fields

class Tool:
    type: Literal["client_side_function", "custom"]
    # ... access based on type
```

**File:** `clients/python/tensorzero/generated_types.py`

Verify dataclass generation creates proper Python types.

### 7.4 TypeScript Types

**File:** `internal/tensorzero-node/lib/bindings/`

Verify generated `.ts` files for:
- `CustomTool.ts`
- `CustomToolFormat.ts`
- `GrammarDefinition.ts`
- `GrammarSyntax.ts`
- Updated `Tool.ts`
- Updated `DynamicToolParams.ts`

---

## Phase 8: Testing

### 8.1 Unit Tests

**File:** `tensorzero-core/src/tool.rs` (add to test module)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // DynamicTool Deserialization Tests
    // ============================================================================

    #[test]
    fn test_dynamic_tool_deserialize_tagged_function() {
        let json = r#"{
            "type": "client_side_function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object"},
            "strict": false
        }"#;

        let dynamic_tool: DynamicTool = serde_json::from_str(json).unwrap();
        assert!(matches!(dynamic_tool.0, Tool::ClientSideFunction(_)));
    }

    #[test]
    fn test_dynamic_tool_deserialize_untagged_function() {
        // Legacy format without type tag
        let json = r#"{
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object"}
        }"#;

        let dynamic_tool: DynamicTool = serde_json::from_str(json).unwrap();
        assert!(matches!(dynamic_tool.0, Tool::ClientSideFunction(_)));
    }

    #[test]
    fn test_dynamic_tool_deserialize_custom_text() {
        let json = r#"{
            "type": "custom",
            "name": "my_tool",
            "description": "A custom tool",
            "format": {
                "type": "text"
            }
        }"#;

        let dynamic_tool: DynamicTool = serde_json::from_str(json).unwrap();
        match dynamic_tool.0 {
            Tool::Custom(custom) => {
                assert_eq!(custom.name, "my_tool");
                assert!(matches!(custom.format, Some(CustomToolFormat::Text)));
            }
            _ => panic!("Expected custom tool"),
        }
    }

    #[test]
    fn test_dynamic_tool_deserialize_custom_grammar_lark() {
        let json = r#"{
            "type": "custom",
            "name": "parser",
            "format": {
                "type": "grammar",
                "grammar": {
                    "syntax": "lark",
                    "definition": "start: WORD+"
                }
            }
        }"#;

        let dynamic_tool: DynamicTool = serde_json::from_str(json).unwrap();
        match dynamic_tool.0 {
            Tool::Custom(custom) => {
                match custom.format {
                    Some(CustomToolFormat::Grammar { grammar }) => {
                        assert_eq!(grammar.syntax, GrammarSyntax::Lark);
                        assert_eq!(grammar.definition, "start: WORD+");
                    }
                    _ => panic!("Expected grammar format"),
                }
            }
            _ => panic!("Expected custom tool"),
        }
    }

    #[test]
    fn test_dynamic_tool_deserialize_custom_grammar_regex() {
        let json = r#"{
            "type": "custom",
            "name": "regex_parser",
            "format": {
                "type": "grammar",
                "grammar": {
                    "syntax": "regex",
                    "definition": "[a-zA-Z]+"
                }
            }
        }"#;

        let dynamic_tool: DynamicTool = serde_json::from_str(json).unwrap();
        match dynamic_tool.0 {
            Tool::Custom(custom) => {
                match custom.format {
                    Some(CustomToolFormat::Grammar { grammar }) => {
                        assert_eq!(grammar.syntax, GrammarSyntax::Regex);
                        assert_eq!(grammar.definition, "[a-zA-Z]+");
                    }
                    _ => panic!("Expected grammar format"),
                }
            }
            _ => panic!("Expected custom tool"),
        }
    }

    #[test]
    fn test_tool_roundtrip_serialization_custom_text() {
        let custom_tool = Tool::Custom(CustomTool {
            name: "test".to_string(),
            description: Some("desc".to_string()),
            format: Some(CustomToolFormat::Text),
        });

        let json = serde_json::to_string(&custom_tool).unwrap();
        let deserialized: Tool = serde_json::from_str(&json).unwrap();

        assert_eq!(custom_tool, deserialized);
    }

    #[test]
    fn test_tool_roundtrip_serialization_custom_grammar() {
        let custom_tool = Tool::Custom(CustomTool {
            name: "parser".to_string(),
            description: None,
            format: Some(CustomToolFormat::Grammar {
                grammar: GrammarDefinition {
                    syntax: GrammarSyntax::Lark,
                    definition: "start: WORD+".to_string(),
                },
            }),
        });

        let json = serde_json::to_string(&custom_tool).unwrap();
        let deserialized: Tool = serde_json::from_str(&json).unwrap();

        assert_eq!(custom_tool, deserialized);
    }

    // ============================================================================
    // Iterator Filtering Tests
    // ============================================================================

    #[test]
    fn test_tools_available_filters_custom_tools() {
        // Create a ToolCallConfig with mix of function and custom tools
        let function_tool = /* ... */;
        let custom_tool = /* ... */;

        let config = ToolCallConfig {
            dynamic_tools_available: vec![function_tool, custom_tool],
            // ...
        };

        // tools_available() should only return function tools
        let tools: Vec<_> = config.tools_available().collect();
        assert_eq!(tools.len(), 1);
        assert!(tools[0].is_function());
    }

    #[test]
    fn test_all_tools_including_custom() {
        // Create a ToolCallConfig with mix of function and custom tools
        let function_tool = /* ... */;
        let custom_tool = /* ... */;

        let config = ToolCallConfig {
            dynamic_tools_available: vec![function_tool, custom_tool],
            // ...
        };

        // all_tools_including_custom() should return both
        let tools: Vec<_> = config.all_tools_including_custom().collect();
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn test_strict_tools_available_filters_custom() {
        // Create config with allowed_tools set
        let function_tool = /* ... name: "func1" */;
        let custom_tool = /* ... name: "custom1" */;

        let config = ToolCallConfig {
            dynamic_tools_available: vec![function_tool, custom_tool],
            allowed_tools: AllowedTools::AllAllowedTools(vec!["func1".into(), "custom1".into()]),
            // ...
        };

        // strict_tools_available() should filter out custom even if in allowed list
        let tools: Vec<_> = config.strict_tools_available().collect();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "func1");
    }

    #[test]
    fn test_strict_tools_including_custom() {
        // Create config with allowed_tools set
        let function_tool = /* ... name: "func1" */;
        let custom_tool = /* ... name: "custom1" */;
        let other_function = /* ... name: "func2" */;

        let config = ToolCallConfig {
            dynamic_tools_available: vec![function_tool, custom_tool, other_function],
            allowed_tools: AllowedTools::AllAllowedTools(vec!["func1".into(), "custom1".into()]),
            // ...
        };

        // strict_tools_including_custom() should return only allowed tools
        let tools: Vec<_> = config.strict_tools_including_custom().collect();
        assert_eq!(tools.len(), 2);
        // Should have func1 and custom1, but not func2
    }

    // ============================================================================
    // Tool Helper Method Tests
    // ============================================================================

    #[test]
    fn test_tool_is_custom() {
        let custom = Tool::Custom(CustomTool {
            name: "test".into(),
            description: None,
            format: None,
        });
        assert!(custom.is_custom());
        assert!(!custom.is_function());
    }

    #[test]
    fn test_tool_is_function() {
        let function = Tool::ClientSideFunction(ClientSideFunctionTool {
            name: "test".into(),
            description: "desc".into(),
            parameters: serde_json::json!({}),
            strict: false,
        });
        assert!(function.is_function());
        assert!(!function.is_custom());
    }
}
```

### 8.2 E2E Tests with OpenAI

**File:** `tensorzero-core/tests/e2e/providers/openai.rs`

```rust
#[tokio::test]
async fn test_openai_custom_tool_text_format() {
    let episode_id = Uuid::now_v7();

    let params = json!({
        "function_name": "basic_test",
        "variant_name": "variant",
        "episode_ids": [episode_id],
        "input": {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{"role": "user", "content": "Test custom tools"}],
        },
        "stream": false,
        "additional_tools": [
            {
                "type": "custom",
                "name": "text_processor",
                "description": "Process text",
                "format": {"type": "text"}
            }
        ]
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&params)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_openai_custom_tool_grammar_lark() {
    let episode_id = Uuid::now_v7();

    let params = json!({
        "function_name": "basic_test",
        "variant_name": "variant",
        "episode_ids": [episode_id],
        "input": {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{"role": "user", "content": "Test grammar tools"}],
        },
        "stream": false,
        "additional_tools": [
            {
                "type": "custom",
                "name": "grammar_parser",
                "format": {
                    "type": "grammar",
                    "grammar": {
                        "syntax": "lark",
                        "definition": "start: WORD+"
                    }
                }
            }
        ]
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&params)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_openai_custom_tool_grammar_regex() {
    let episode_id = Uuid::now_v7();

    let params = json!({
        "function_name": "basic_test",
        "variant_name": "variant",
        "episode_ids": [episode_id],
        "input": {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{"role": "user", "content": "Test regex tools"}],
        },
        "stream": false,
        "additional_tools": [
            {
                "type": "custom",
                "name": "regex_parser",
                "format": {
                    "type": "grammar",
                    "grammar": {
                        "syntax": "regex",
                        "definition": "[0-9]+"
                    }
                }
            }
        ]
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&params)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_openai_mixed_function_and_custom_tools() {
    let episode_id = Uuid::now_v7();

    let params = json!({
        "function_name": "basic_test",
        "variant_name": "variant",
        "episode_ids": [episode_id],
        "input": {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{"role": "user", "content": "Test mixed tools"}],
        },
        "stream": false,
        "additional_tools": [
            {
                // Legacy untagged format (function tool)
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            },
            {
                // New tagged custom format
                "type": "custom",
                "name": "text_processor",
                "format": {"type": "text"}
            }
        ]
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&params)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_openai_custom_tool_with_allowed_tools() {
    let episode_id = Uuid::now_v7();

    let params = json!({
        "function_name": "basic_test",
        "variant_name": "variant",
        "episode_ids": [episode_id],
        "input": {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{"role": "user", "content": "Test"}],
        },
        "stream": false,
        "additional_tools": [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object"}
            },
            {
                "type": "custom",
                "name": "custom_tool",
                "format": {"type": "text"}
            }
        ],
        "allowed_tools": ["custom_tool"]  // Only allow the custom tool
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&params)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    // TODO: Verify only custom_tool was sent to OpenAI
}
```

### 8.3 Test Custom Tools Filtered for Non-OpenAI Providers

**File:** `tensorzero-core/tests/e2e/providers/anthropic.rs`

```rust
#[tokio::test]
async fn test_anthropic_ignores_custom_tools() {
    let episode_id = Uuid::now_v7();

    let params = json!({
        "function_name": "basic_test",
        "variant_name": "anthropic_variant",  // Use Anthropic model
        "episode_ids": [episode_id],
        "input": {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{"role": "user", "content": "Test"}],
        },
        "stream": false,
        "additional_tools": [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "type": "custom",
                "name": "custom_tool",
                "format": {"type": "text"}
            }
        ]
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&params)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    // Custom tool should be automatically filtered by tools_available()
    // Only get_weather should be sent to Anthropic
}
```

### 8.4 Database Round-Trip Tests

**File:** `tensorzero-core/tests/e2e/db/` or create new test file

```rust
#[tokio::test]
async fn test_custom_tool_storage_and_retrieval() {
    // Create tool config with custom tool
    let custom_tool = Tool::Custom(CustomTool {
        name: "test_tool".to_string(),
        description: Some("Test".to_string()),
        format: Some(CustomToolFormat::Text),
    });

    let tool_config = ToolCallConfig {
        dynamic_tools_available: vec![/* custom_tool wrapped in ToolConfig */],
        // ... other fields
    };

    // Convert to database insert format
    let db_insert = tool_config.to_database_insert();

    // Serialize to JSON (simulating DB storage)
    let json = serde_json::to_string(&db_insert).unwrap();

    // Deserialize from JSON (simulating DB retrieval)
    let retrieved_insert: ToolCallConfigDatabaseInsert =
        serde_json::from_str(&json).unwrap();

    // Convert back to ToolCallConfig
    let retrieved_config = retrieved_insert.into_tool_call_config(&config, &tools);

    // Verify custom tool is preserved
    assert!(retrieved_config.all_tools_including_custom().any(|t| {
        matches!(t, ToolConfig::Dynamic(d) if matches!(d.tool, Tool::Custom(_)))
    }));

    // Verify it's filtered from default iterator
    assert!(!retrieved_config.tools_available().any(|t| {
        matches!(t, ToolConfig::Dynamic(d) if matches!(d.tool, Tool::Custom(_)))
    }));
}
```

### 8.5 Python Client Tests

**File:** `clients/python/tests/test_custom_tools.py` (new file)

```python
import pytest
from tensorzero import TensorZeroGateway

def test_custom_tool_text_format(gateway_url):
    """Test creating and using a custom tool with text format"""
    client = TensorZeroGateway(gateway_url)

    response = client.inference(
        function_name="test_function",
        variant_name="test_variant",
        episode_ids=["test-episode"],
        input={
            "messages": [{"role": "user", "content": "Test"}]
        },
        additional_tools=[
            {
                "type": "custom",
                "name": "text_processor",
                "description": "Process text",
                "format": {"type": "text"}
            }
        ]
    )

    assert response.status_code == 200

def test_custom_tool_grammar_format(gateway_url):
    """Test custom tool with grammar format"""
    client = TensorZeroGateway(gateway_url)

    response = client.inference(
        function_name="test_function",
        variant_name="test_variant",
        episode_ids=["test-episode"],
        input={
            "messages": [{"role": "user", "content": "Test"}]
        },
        additional_tools=[
            {
                "type": "custom",
                "name": "grammar_parser",
                "format": {
                    "type": "grammar",
                    "grammar": {
                        "syntax": "lark",
                        "definition": "start: WORD+"
                    }
                }
            }
        ]
    )

    assert response.status_code == 200

def test_mixed_tools_backward_compatibility(gateway_url):
    """Test mixing legacy untagged and new tagged tool formats"""
    client = TensorZeroGateway(gateway_url)

    response = client.inference(
        function_name="test_function",
        variant_name="test_variant",
        episode_ids=["test-episode"],
        input={
            "messages": [{"role": "user", "content": "Test"}]
        },
        additional_tools=[
            # Legacy format (untagged function tool)
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object"}
            },
            # New format (tagged function tool)
            {
                "type": "client_side_function",
                "name": "get_time",
                "description": "Get time",
                "parameters": {"type": "object"}
            },
            # New format (custom tool)
            {
                "type": "custom",
                "name": "custom_tool",
                "format": {"type": "text"}
            }
        ]
    )

    assert response.status_code == 200
```

---

## Implementation Checklist

### Phase 1: Core Types
- [ ] Add `CustomTool`, `CustomToolFormat`, `GrammarDefinition`, `GrammarSyntax` structs
- [ ] Add PyO3 methods for `CustomTool`
- [ ] Add `Custom(CustomTool)` variant to `Tool` enum
- [ ] Update `Tool::name()` method
- [ ] Update `Tool::into_dynamic_tool_config()` (handle or document limitation)
- [ ] Update `Tool` PyO3 methods to handle custom tools
- [ ] Add `is_custom()` and `is_function()` helper methods to `Tool`
- [ ] Add `is_custom()` and `is_function()` helper methods to `ToolConfig`
- [ ] Create `DynamicTool` wrapper with custom deserializer
- [ ] Update `DynamicToolParams.additional_tools` type

### Phase 2: Iterator Strategy
- [ ] Update `tools_available()` to filter custom tools
- [ ] Update `strict_tools_available()` to filter custom tools
- [ ] Add `all_tools_including_custom()` method
- [ ] Add `strict_tools_including_custom()` method
- [ ] Document iterator usage clearly

### Phase 3: OpenAI Interface
- [ ] Extend `OpenAICompatibleTool` enum with `Custom` variant
- [ ] Add OpenAI custom tool structures (`OpenAICustomDefinition`, etc.)
- [ ] Update conversion from `OpenAICompatibleTool` to `Tool`
- [ ] Update tool conversion call in `inference()`

### Phase 4: Tool Processing
- [ ] Update `FunctionConfig::prepare_tool_config()` to handle `DynamicTool`
- [ ] Handle custom tools in tool config preparation (may skip validation)
- [ ] Audit `into_dynamic_tool_config()` call sites

### Phase 5: Providers
- [ ] Update OpenAI provider to use `all_tools_including_custom()`
- [ ] Add OpenAI custom tool output structures
- [ ] Update OpenAI tool conversion to handle custom tools
- [ ] **Verify other providers automatically filter** (no changes needed)

### Phase 6: Storage
- [ ] Verify `ToolCallConfigDatabaseInsert` handles custom tools
- [ ] Test serialization/deserialization
- [ ] Test database round-trip

### Phase 7: Bindings
- [ ] Run `cargo test export_bindings`
- [ ] Run `pnpm build-bindings` in tensorzero-node
- [ ] Verify Python types in `generated_types.py`
- [ ] Verify TypeScript types in `lib/bindings/`
- [ ] Update Python type hints in `.pyi` file if needed

### Phase 8: Testing
- [ ] Unit tests for `DynamicTool` deserializer (all formats)
- [ ] Unit tests for custom tool serialization
- [ ] Unit tests for iterator filtering
- [ ] Unit tests for tool helper methods
- [ ] E2E: OpenAI with text format custom tool
- [ ] E2E: OpenAI with grammar format (lark and regex)
- [ ] E2E: Mixed function and custom tools
- [ ] E2E: Custom tools with allowed_tools
- [ ] E2E: Custom tools filtered for Anthropic
- [ ] Database round-trip test
- [ ] Python client tests (all formats)

### Phase 9: Documentation
- [ ] Update tool-use guide with custom tools examples
- [ ] Update API reference docs
- [ ] Document iterator strategy
- [ ] Add migration guide for custom tools

---

## Key Files Reference

### Core Tool System
- `tensorzero-core/src/tool.rs` - **MAIN FILE** - Tool types, ToolCallConfig, iterators
  - Lines 66-68: Tool enum (add Custom variant)
  - Lines 78-82: Tool::name() (update)
  - Lines 347-356: ToolCallConfig struct
  - Lines 400+: Iterator methods (update all)
  - Lines 650-660: ToolCallConfigDatabaseInsert
  - Lines 1255-1268: DynamicToolParams (update additional_tools type)

### API Endpoints
- `tensorzero-core/src/endpoints/openai_compatible.rs` - OpenAI compatible interface
  - Lines 387-395: OpenAICompatibleTool enum (extend)
  - Lines 776-781: Tool conversion (update)
  - Lines 1177-1192: From impl (replace)

### Tool Processing
- `tensorzero-core/src/function.rs` - Tool config preparation
  - Line 342: prepare_tool_config() (update for DynamicTool)

### Providers
- `tensorzero-core/src/providers/openai/mod.rs` - **ONLY PROVIDER TO CHANGE**
  - Lines 1520-1563: prepare_openai_tools() (use new iterator)
  - Lines 2000+: Tool conversion (add custom tool support)

### Tests
- `tensorzero-core/tests/e2e/providers/openai.rs` - OpenAI E2E tests
- `tensorzero-core/tests/e2e/providers/anthropic.rs` - Verify filtering
- `tensorzero-core/tests/e2e/inference/tool_params.rs` - Tool params tests
- `clients/python/tests/` - Python client tests

### Bindings
- `internal/tensorzero-node/lib/bindings/` - TypeScript bindings (generated)
- `clients/python/tensorzero/generated_types.py` - Python bindings (generated)

---

## Architecture Decisions

### 1. Iterator Strategy (CONFIRMED)

**Decision:** Existing iterators filter custom tools by default. New iterator includes them.

**Rationale:**
- ✅ Backward compatible (existing code unaffected)
- ✅ No need to modify all providers
- ✅ Explicit opt-in for custom tools
- ✅ Clear API semantics

**Implementation:**
```rust
// Default - filters custom tools
tool_config.tools_available()
tool_config.strict_tools_available()

// OpenAI only - includes custom tools
tool_config.all_tools_including_custom()
tool_config.strict_tools_including_custom()
```

### 2. DynamicTool Deserializer Strategy

**Decision:** Custom deserializer tries tagged format first, falls back to untagged.

**Rationale:**
- ✅ Maintains backward compatibility with existing untagged ClientSideFunctionTool
- ✅ Supports new tagged format for both function and custom tools
- ✅ Single field for all tool types

### 3. Custom Tool Parameter Validation

**Decision:** Custom tools skip parameter validation (no JSON schema).

**Rationale:**
- Custom tools don't have traditional JSON schema parameters
- Validation happens at OpenAI's API level
- `into_dynamic_tool_config()` should not be called for custom tools

**Implementation:** Use `unreachable!()` in `into_dynamic_tool_config()` for custom tools, audit call sites.

### 4. Storage Format

**Decision:** Normalize to internal `Tool` enum for storage.

**Rationale:**
- ✅ Already have `Vec<Tool>` in database schema
- ✅ Consistent with current architecture
- ✅ No coupling to OpenAI's API format

### 5. Provider Handling

**Decision:** Only OpenAI receives custom tools, others automatically filter.

**Rationale:**
- No other provider supports custom tools currently
- Iterator strategy makes this automatic
- Easy to extend if other providers add support

---

## Open Questions / Design Decisions

### 1. ToolConfig Structure for Custom Tools

**Question:** How should custom tools be represented in `ToolConfig`?

**Current structure:**
```rust
pub enum ToolConfig {
    Static(Arc<StaticToolConfig>),
    Dynamic(DynamicToolConfig),
}
```

**Need to investigate:** Can `DynamicToolConfig` handle tools without parameters? Or do we need a separate variant?

**Possible solution:** Make `DynamicToolConfig` parameters optional or add pattern matching in processing.

### 2. Custom Tool Results

**Question:** When OpenAI returns a custom tool call result, what format is it? Does TensorZero need special handling?

**Action:** Research OpenAI's response format for custom tool calls. May be out of scope for this PR (can handle in follow-up).

### 3. Error Handling for Invalid Grammar Syntax

**Question:** What if OpenAI sends an unknown grammar syntax (not "lark" or "regex")?

**Decision:** Default to "lark" with debug log warning. Non-breaking.

### 4. Allowed Tools with FunctionDefault Mode

**Question:** In `strict_tools_including_custom()`, should `FunctionDefault` mode include custom tools?

**Current decision:** No - `FunctionDefault` means "use function config tools" which are function-based. Only `AllAllowedTools`/`DynamicAllowedTools` can include custom.

---

## Success Metrics

1. ✅ **Backward Compatibility:** All existing tests pass without modification
2. ✅ **New Format Support:** Can send custom tools in tagged format
3. ✅ **Legacy Support:** Can still send untagged function tools
4. ✅ **Provider Isolation:** Custom tools automatically filtered for non-OpenAI providers
5. ✅ **Storage Integrity:** Custom tools round-trip through database correctly
6. ✅ **Type Safety:** Python and TypeScript types correctly generated
7. ✅ **Documentation:** Clear examples and migration guide

---

## Future Work (Out of Scope)

- Static custom tools (defined in TensorZero config)
- MCP (Model Context Protocol) tool integration
- Custom tool call result handling/validation
- Support custom tools in other providers (when they add support)
- UI support for custom tools in playground
- Advanced grammar validation
- Custom tool analytics/metrics

---

## Timeline Estimate

- Phase 1 (Core Types): 4 hours
- Phase 2 (Iterator Strategy): 2 hours
- Phase 3 (OpenAI Interface): 2 hours
- Phase 4 (Tool Processing): 2 hours
- Phase 5 (OpenAI Provider): 3 hours
- Phase 6 (Storage Verification): 1 hour
- Phase 7 (Bindings): 1 hour
- Phase 8 (Testing): 5 hours
- Phase 9 (Documentation): 2 hours

**Total:** ~22 hours

---

## Related Context

- Current branch: `viraj/mandatory-dynamic-allowed-tools`
- Recent work: `allowed_tools` enforcement (completed)
- Related issue: Tool enum extensibility (comment at tool.rs:50-61 anticipated this)

---

## Notes

- The iterator strategy is elegant and requires minimal changes to existing code
- Only OpenAI provider needs updates - all others automatically work correctly
- The `Tool` enum was designed to be extensible (this was anticipated)
- Database storage already uses `Vec<Tool>` so custom tools should "just work"
- Biggest complexity is in the custom deserializer for backward compatibility
- Need to audit `into_dynamic_tool_config()` call sites to ensure custom tools are filtered before reaching parameter validation
