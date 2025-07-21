#!/bin/bash
set -e

echo "🚀 Setting up TensorZero + OpenAI Agents SDK Integration Development Environment"
echo "=============================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "Cargo.toml" ]] || [[ ! -d "tensorzero-core" ]]; then
    print_error "Please run this script from the TensorZero root directory"
    exit 1
fi

print_status "Setting up development environment..."

# 1. TensorZero Python Client with Agents Integration
print_status "Setting up TensorZero Python client with Agents SDK integration..."

cd clients/python

# Handle conda environment conflict
if [[ -n "$CONDA_PREFIX" ]]; then
    print_warning "Conda environment detected. Deactivating conda to avoid conflicts..."
    conda deactivate 2>/dev/null || true
    unset CONDA_PREFIX
fi

if command -v uv &> /dev/null; then
    print_status "Using uv for Python environment..."
    # Use .venv to match maturin's expectations
    uv venv --python 3.11 .venv
    source .venv/bin/activate
    
    # Install TensorZero client in development mode with agents optional dependency
    print_status "Installing TensorZero client with agents support..."
    uv sync --group dev --group agents
    uv run maturin develop --uv
else
    print_status "Using standard Python venv..."
    # Use .venv to match maturin's expectations
    python3 -m venv .venv
    source .venv/bin/activate
    
    # Install TensorZero client in development mode with agents optional dependency
    print_status "Installing TensorZero client with agents support..."
    pip install -e ".[agents]"
    pip install maturin
    maturin develop
fi

print_success "TensorZero Python client with Agents SDK integration ready"

# Go back to root
cd ../..

# 2. Docker Setup for TensorZero
print_status "Setting up Docker environment for TensorZero..."

# Check if Docker is running
if ! docker info &> /dev/null; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Start TensorZero with ClickHouse for testing
print_status "Starting TensorZero Gateway and ClickHouse..."
cd examples/rag-retrieval-augmented-generation/simple-agentic-rag

# Use the existing docker-compose from the TensorZero example
if [[ -f "docker-compose.yml" ]]; then
    docker-compose up -d
    print_success "TensorZero services started"
else
    print_warning "No docker-compose.yml found in simple-agentic-rag directory"
    print_status "You'll need to start TensorZero services manually"
fi

cd ../../..

# 3. Set up example environments
print_status "Setting up example environments..."

# Pure Agents SDK example
print_status "Setting up pure Agents SDK example..."
cd examples/rag-retrieval-augmented-generation/simple-agentic-rag-openai

if command -v uv &> /dev/null; then
    uv venv --python 3.11 .venv-pure
    source .venv-pure/bin/activate
    uv pip install -e .
else
    python3 -m venv .venv-pure
    source .venv-pure/bin/activate
    pip install -e .
fi

print_success "Pure Agents SDK example environment ready"

# Go back to root
cd ../../..

# 4. Test that everything works
print_status "Running basic setup tests..."

# Test TensorZero Python client with agents
print_status "Testing TensorZero client with agents integration..."
cd clients/python
source .venv/bin/activate

if python -c "import tensorzero; print('✅ TensorZero imported successfully')" 2>/dev/null; then
    print_success "TensorZero client working"
else
    print_error "Failed to import TensorZero client"
fi

if python -c "import tensorzero.agents; print('✅ TensorZero Agents integration imported successfully')" 2>/dev/null; then
    print_success "TensorZero Agents integration working"
else
    print_warning "TensorZero Agents integration not available (install with: pip install tensorzero[agents])"
fi

cd ../..

# 5. Build TensorZero (if needed)
if [[ ! -f "target/debug/tensorzero-gateway" ]]; then
    print_status "Building TensorZero gateway..."
    cargo build --bin tensorzero-gateway
    print_success "TensorZero gateway built"
fi

# 6. Final instructions
print_success "Development environment setup complete!"
echo ""
echo "📋 Next Steps:"
echo "=============="
echo ""
echo "1. Set your OpenAI API key:"
echo "   export OPENAI_API_KEY='sk-your-key-here'"
echo ""
echo "2. Test the TensorZero + Agents SDK integration:"
echo "   cd clients/python"
echo "   source .venv/bin/activate"
echo "   python -m pytest tests/test_agents_integration.py -v"
echo ""
echo "3. Try the integration example:"
echo "   cd examples/rag-retrieval-augmented-generation/simple-agentic-rag"
echo "   python integration_example.py"
echo ""
echo "4. Compare with pure Agents SDK example:"
echo "   cd examples/rag-retrieval-augmented-generation/simple-agentic-rag-openai"
echo "   source .venv-pure/bin/activate"
echo "   python main.py"
echo ""
echo "5. TensorZero services should be running at:"
echo "   Gateway: http://localhost:3000"
echo "   ClickHouse: http://localhost:8123"
echo ""
echo "🎯 Integration complete! Use: pip install tensorzero[agents]" 