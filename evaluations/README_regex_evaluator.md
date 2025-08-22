# Regex Evaluator

The regex evaluator is a new addition to TensorZero's static evaluations that allows you to evaluate inference responses using regular expression patterns.

## Overview

The regex evaluator checks if a given regex pattern matches the text content of an inference response. It returns a boolean value indicating whether the pattern was found in the response.

## Configuration

To use the regex evaluator, add it to your evaluation configuration:

```toml
[evaluations.my_evaluation.evaluators.my_regex_evaluator]
type = "regex"
regex = "your_regex_pattern_here"
cutoff = 0.8  # optional
```

### Parameters

- `type`: Must be `"regex"`
- `regex`: The regular expression pattern to match against the response
- `cutoff`: Optional threshold value (default: no cutoff)

## How It Works

1. **Text Extraction**: The evaluator extracts text content from the inference response:
   - For chat responses: Combines text from all content blocks (text, tool calls, thoughts)
   - For JSON responses: Uses the raw JSON output as text

2. **Pattern Matching**: Applies the regex pattern to the extracted text using Rust's `regex` crate

3. **Result**: Returns `true` if the pattern matches, `false` otherwise

## Examples

### Basic Usage

```toml
# Check if response contains a number
[evaluations.my_evaluation.evaluators.contains_number]
type = "regex"
regex = "\\d+"
```

### Common Patterns

```toml
# Check for email addresses
[evaluations.my_evaluation.evaluators.has_email]
type = "regex"
regex = "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"

# Check for URLs
[evaluations.my_evaluation.evaluators.has_url]
type = "regex"
regex = "https?://[^\\s]+"

# Check for JSON structure
[evaluations.my_evaluation.evaluators.is_json]
type = "regex"
regex = "^\\s*\\{.*\\}\\s*$"

# Check for specific keywords
[evaluations.my_evaluation.evaluators.contains_keyword]
type = "regex"
regex = "\\b(important|critical|urgent)\\b"
```

## Error Handling

- **Invalid Regex**: If the regex pattern is invalid, the evaluator will return an error
- **Type Mismatch**: If the datapoint and inference response types don't match, an error is returned

## Testing

The regex evaluator includes comprehensive tests covering:
- Chat response evaluation
- JSON response evaluation
- Invalid regex patterns
- Type mismatches
- Text extraction from different content block types

## Integration

The regex evaluator integrates seamlessly with TensorZero's existing evaluation system:
- Works with both chat and JSON inference responses
- Supports all existing evaluation features (cutoffs, metrics, etc.)
- Follows the same patterns as other evaluators (exact_match, llm_judge)

## Use Cases

- **Content Validation**: Ensure responses contain required information
- **Format Checking**: Verify responses follow expected patterns
- **Quality Assurance**: Check for specific keywords or structures
- **Compliance**: Ensure responses meet regulatory or business requirements
- **Data Extraction**: Validate that responses contain extractable data

## Limitations

- Only supports boolean results (true/false)
- Pattern matching is case-sensitive by default
- Complex regex patterns may impact performance
- Does not support capturing groups or complex regex features beyond basic matching
