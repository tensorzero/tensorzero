# Tree Export Files for D3.js Visualization

This directory contains JSON files exported from the integration tests in `parsing.rs`, ready for use with D3.js tree visualizations.

## Files Generated

### Tree JSON Files (14 total)
- **Rust examples**: `rust_simple_function.json`, `rust_struct_with_impl.json`, `rust_complex_function.json`, `rust_loop_example.json`, `rust_data_structures.json`, `rust_error_handling.json`
- **TypeScript examples**: `typescript_simple_function.json`, `typescript_class_with_methods.json`
- **Python examples**: `python_simple_function.json`, `python_class_with_methods.json`
- **TOML examples**: `toml_simple_config.json`, `toml_complex_config.json`
- **Markdown examples**: `markdown_simple.json`, `markdown_with_lists.json`

### Index and Viewer
- **`index.json`**: Master index of all trees with metadata
- **`d3_tree_viewer.html`**: Interactive HTML viewer for the trees
- **`README.md`**: This documentation file

## JSON Structure

Each tree JSON file contains:

```json
{
  "metadata": {
    "name": "rust_simple_function",
    "language": "rust",
    "description": "Simple Rust function with println! macro",
    "source_length": 51,
    "node_count": 22
  },
  "source_code": "fn hello_world() {\n    println!(\"Hello, world!\");\n}",
  "tree": {
    "name": "source_file",
    "text": "",
    "full_text": "...",
    "start_position": {"row": 0, "column": 0},
    "end_position": {"row": 2, "column": 1},
    "is_leaf": false,
    "child_count": 1,
    "children": [...]
  }
}
```

### Tree Node Properties
- **`name`**: AST node type (e.g., "function_item", "identifier")
- **`text`**: Source text for leaf nodes only
- **`full_text`**: Complete source text span for this node
- **`start_position`/`end_position`**: Line/column positions in source
- **`is_leaf`**: Boolean indicating if node has children
- **`child_count`**: Number of direct children
- **`children`**: Array of child nodes (recursive structure)

## Usage

### View with HTML Interface

**🚀 Quick Start (Recommended):**
```bash
./start_server
```
This will automatically:
- Start a local web server with CORS support
- Open your browser to the tree viewer
- Display all available trees for selection

**Manual Method:**
If the quick start doesn't work, you can:
1. Start any web server in this directory (needed for CORS)
2. Open `http://localhost:8000/d3_tree_viewer.html` in your browser

**Features:**
- Select a tree from the dropdown
- Choose different layout options (tree, cluster, radial)
- Hover over nodes for detailed information
- Use mouse wheel to zoom, drag to pan

### Integrate with D3.js
```javascript
// Load a tree
d3.json('rust_simple_function.json').then(function(data) {
    // Convert to D3 hierarchy
    const root = d3.hierarchy(data.tree);

    // Create tree layout
    const treeLayout = d3.tree().size([width, height]);
    treeLayout(root);

    // Render nodes and links...
});
```

### Load Index for Tree Selection
```javascript
d3.json('index.json').then(function(index) {
    index.trees.forEach(tree => {
        console.log(`${tree.name}: ${tree.node_count} nodes, ${tree.language}`);
    });
});
```

## Tree Statistics

| Language   | Trees | Avg Nodes | Total Nodes |
|------------|-------|-----------|-------------|
| Rust       | 6     | 44        | 263         |
| TypeScript | 2     | 50        | 99          |
| Python     | 2     | 54        | 107         |
| TOML       | 2     | 47        | 94          |
| Markdown   | 2     | 49        | 97          |
| **Total**  | **14**| **47**    | **660**     |

## Examples by Complexity

### Simple Examples (< 30 nodes)
- `rust_simple_function.json` (22 nodes) - Basic function with macro
- `python_simple_function.json` (19 nodes) - Arithmetic function

### Medium Examples (30-60 nodes)
- `rust_struct_with_impl.json` (53 nodes) - Struct with implementation
- `typescript_simple_function.json` (32 nodes) - Function with template strings
- `rust_loop_example.json` (36 nodes) - For loop with range

### Complex Examples (60+ nodes)
- `typescript_class_with_methods.json` (67 nodes) - Class with private fields
- `python_class_with_methods.json` (88 nodes) - Class with decorators
- `rust_complex_function.json` (59 nodes) - If-else conditional logic

## Generated From

These trees were exported from integration tests in:
```
src/parsing.rs
```

The tests use tree-sitter parsers for multiple languages and contain comprehensive examples of real-world code structures, making them excellent for visualization and analysis purposes.

## Regenerating Files

To regenerate the JSON files:

```bash
cd /path/to/cursorzero-tests/examples/integrations/cursor/experimental
cargo run --bin export_trees
```

This will overwrite all JSON files with fresh data from the parsing tests.
