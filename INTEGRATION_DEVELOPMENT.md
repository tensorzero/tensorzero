# TensorZero + OpenAI Agents SDK Integration Development

This guide helps you set up and develop the integration between TensorZero and the OpenAI Agents SDK.

## 🎯 Project Goal

Create a seamless integration that allows developers to use TensorZero's production-grade LLM infrastructure with the OpenAI Agents SDK's intuitive agent abstractions.

### Target Developer Experience

```python
# What developers can now write:
# pip install tensorzero[agents]

from agents import Agent, Runner
import tensorzero.agents as tz_agents

# One-line setup - automatically detects templates and tools from tensorzero.toml
await tz_agents.setup_tensorzero_agents("config/tensorzero.toml")

# Option 1: Auto-create agent from TensorZero function
agent = tz_agents.create_agent_from_tensorzero_function(
    function_name="multi_hop_rag_agent",
    variant_name="baseline",
    tools=[my_tools...]
)

# Option 2: Manual agent creation - templates still work automatically
agent = Agent(
    name="RAG Agent", 
    model="tensorzero::function_name::multi_hop_rag_agent::baseline",
    tools=[my_tools...]
)

result = await Runner(agent=agent).run("What's the weather like?")
# ✅ Templates applied automatically
# ✅ TensorZero observability logged to ClickHouse
# ✅ Episodes tracked automatically
# ✅ A/B testing works with variants
# ✅ All TensorZero production features available
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- OpenAI API key

### Setup Development Environment

```bash
# 1. Clone and setup
git clone <tensorzero-repo>
cd tensorzero

# 2. Run the setup script (sets up everything automatically)
./dev_setup.sh

# 3. Set your OpenAI API key
export OPENAI_API_KEY='sk-your-key-here'

# 4. Test the TensorZero + Agents SDK integration
make demo-integration
```

Or use the Makefile for individual steps:

```bash
make help                   # Show all available commands
make quick-setup            # Complete setup + demo
make install-agents         # Install tensorzero[agents] in current env
make test-integration       # Test TensorZero + Agents SDK integration
make docker-up              # Start TensorZero services
make show-usage             # Show integration usage examples
```

## 📁 Project Structure

```
tensorzero/
├── clients/python/                      # TensorZero Python client
│   ├── tensorzero/
│   │   ├── agents.py                    # 🆕 Agents SDK integration module
│   │   ├── __init__.py                  # Exports agents functionality
│   │   └── ...                          # Other TensorZero client code
│   ├── tests/
│   │   ├── test_agents_integration.py   # 🆕 Integration tests
│   │   └── ...                          # Other tests
│   ├── pyproject.toml                   # 🆕 Added [agents] dependency group
│   └── .venv/                           # Single dev environment
├── examples/rag-retrieval-augmented-generation/
│   ├── simple-agentic-rag/              # Original TensorZero version
│   │   ├── integration_example.py       # 🆕 Integration example
│   │   └── ...                          # Original files
│   └── simple-agentic-rag-openai/       # Pure Agents SDK version (for comparison)
│       ├── main.py                      # Pure implementation
│       ├── tools.py                     # Function tools
│       ├── test_comparison.py           # Comparison tests
│       └── .venv-pure/                  # Reference environment
├── dev_setup.sh                        # Updated setup script
├── Makefile.integration                # Updated development commands
└── INTEGRATION_DEVELOPMENT.md          # This file
```

### 🆕 Key Changes from Separate Package Approach

✅ **Single Package**: Integration is now `tensorzero.agents` (not separate `tensorzero-agents`)  
✅ **Simple Installation**: `pip install tensorzero[agents]` (not separate package)  
✅ **Single Environment**: All development in `clients/python/.venv`  
✅ **Cleaner Architecture**: Uses TensorZero's existing OpenAI compatibility  
✅ **Automatic Features**: All TensorZero production features work transparently

## 🧪 Development Workflow

### 1. Test Pure Agents SDK Version

The pure version serves as our target developer experience:

```bash
cd examples/rag-retrieval-augmented-generation/simple-agentic-rag-openai
source .venv-agents/bin/activate
python main.py
```

### 2. Work on Integration

```bash
source .venv-integration/bin/activate
cd tensorzero_agents_integration

# Start implementing the integration
# See "Integration Architecture" section below
```

### 3. Compare Implementations

```bash
make test-pure  # Test pure Agents SDK version
# Eventually: make test-integration
```

## 🏗️ Integration Architecture

Based on our research, the integration uses TensorZero's built-in OpenAI compatibility:

### Core Approach

1. **Leverage OpenAI Compatibility**: TensorZero already provides OpenAI-compatible endpoints
2. **Smart Client Patching**: Patch the OpenAI client to auto-detect TensorZero functions
3. **Template Auto-Conversion**: Automatically convert normal messages to `tensorzero::arguments` format
4. **Configuration Parsing**: Use TensorZero's config parser to detect templated functions

### Key Components

```python
# tensorzero_agents/integration.py
async def setup_tensorzero_agents(config_file: str):
    """Main setup function that patches Agents SDK to use TensorZero"""
    
    # 1. Parse TensorZero config to detect templated functions
    template_detector = TensorZeroTemplateDetector(config_file)
    
    # 2. Set up TensorZero's OpenAI-compatible client
    client = await patch_openai_client(AsyncOpenAI(), config_file=config_file)
    
    # 3. Add smart template conversion
    smart_client = patch_client_with_template_detection(client, template_detector)
    
    # 4. Set as default for Agents SDK
    set_default_openai_client(smart_client)
```

## 🧬 Implementation Details

### Template Detection

The integration automatically detects which TensorZero functions use templates:

```python
class TensorZeroTemplateDetector:
    def __init__(self, config_file: str):
        # Parse tensorzero.toml
        self.config = toml.load(config_file)
        self.templated_functions = self._detect_templated_functions()
    
    def is_templated_function(self, model_name: str) -> bool:
        # Check if model name like "tensorzero::function_name::my_function" 
        # refers to a function with templates
```

### Message Conversion

When a templated function is detected, messages are automatically converted:

```python
# User writes normal Agents SDK code:
messages = [{"role": "user", "content": "What's the weather?"}]

# Integration automatically converts to:
messages = [
    {
        "role": "user", 
        "content": [
            {
                "type": "text",
                "tensorzero::arguments": {
                    "user_input": "What's the weather?",
                    "timestamp": "2024-01-15T10:30:00Z",
                    # ... other extracted variables
                }
            }
        ]
    }
]
```

### Tool Loading

Tools are automatically loaded from TensorZero configuration:

```python
def load_tensorzero_tools(config_file: str, function_name: str):
    """Convert TensorZero tools to Agents SDK function_tool format"""
    config = toml.load(config_file)
    function_config = config['functions'][function_name]
    
    tools = []
    for tool_name in function_config.get('tools', []):
        tool_config = config['tools'][tool_name]
        # Convert to OpenAI tool format that Agents SDK understands
        tools.append(convert_tool_config(tool_config))
    
    return tools
```

## 🧪 Testing Strategy

### Test Phases

1. **Pure Agents SDK**: Verify the target experience works
2. **TensorZero Integration**: Test that integration provides same UX + TensorZero benefits
3. **Feature Parity**: Ensure all TensorZero features work through integration
4. **Performance**: Verify no significant performance regression

### Test Files

```bash
examples/rag-retrieval-augmented-generation/simple-agentic-rag-openai/test_comparison.py
tensorzero_agents_integration/tests/test_integration.py
tensorzero_agents_integration/tests/test_template_detection.py
tensorzero_agents_integration/tests/test_tool_conversion.py
```

## 🌟 Features to Implement

### Phase 1: Basic Integration
- [x] Pure Agents SDK implementation ✅
- [ ] TensorZero config parsing
- [ ] Template detection
- [ ] Automatic message conversion
- [ ] Tool loading

### Phase 2: Advanced Features  
- [ ] Session/episode management
- [ ] A/B testing support
- [ ] Streaming support
- [ ] Error handling
- [ ] Performance optimization

### Phase 3: Production Features
- [ ] Comprehensive testing
- [ ] Documentation
- [ ] Examples
- [ ] Performance benchmarks

## 📚 Reference Implementation

See `examples/rag-retrieval-augmented-generation/simple-agentic-rag-openai/` for the pure Agents SDK implementation that serves as our target.

### Key Differences

| Aspect | TensorZero Original | Pure Agents SDK | Target Integration |
|--------|-------------------|-----------------|-------------------|
| Tool Loop | Manual | Built-in Runner.run() | Built-in Runner.run() |
| Tools | JSON schemas | @function_tool | Auto-converted |
| Templates | Automatic | Manual strings | Automatic |
| Config | tensorzero.toml | Python code | tensorzero.toml |
| Observability | Automatic | None | Automatic |

## 🤝 Contributing

1. Set up development environment: `./dev_setup.sh`
2. Work in the `tensorzero_agents_integration/` directory
3. Test with: `make test-integration`
4. Follow the architecture outlined above

## 🔗 Key Files in TensorZero Codebase

- `tensorzero-core/src/config_parser/mod.rs` - Configuration parsing
- `tensorzero-core/src/endpoints/openai_compatible.rs` - OpenAI compatibility
- `clients/python/src/lib.rs` - Python client implementation
- `examples/rag-retrieval-augmented-generation/simple-agentic-rag/` - Original example

The integration leverages TensorZero's existing infrastructure rather than reimplementing features. 