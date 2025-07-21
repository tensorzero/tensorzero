#!/bin/bash
set -e

echo "🔧 Fixing TensorZero + Agents SDK Development Environment"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

print_status "Cleaning up existing environments..."

# Remove any existing problematic virtual environments
rm -rf clients/python/.venv-dev
rm -rf clients/python/.venv
rm -rf examples/rag-retrieval-augmented-generation/simple-agentic-rag-openai/.venv-agents

print_success "Cleaned up old environments"

# Handle conda environment conflict
if [[ -n "$CONDA_PREFIX" ]]; then
    print_warning "Conda environment detected. Please run:"
    echo "  conda deactivate"
    echo "  unset CONDA_PREFIX"
    echo "  ./fix_dev_setup.sh"
    exit 1
fi

# Ensure we're not in any virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    print_warning "Virtual environment active. Deactivating..."
    deactivate 2>/dev/null || true
    unset VIRTUAL_ENV
fi

print_status "Setting up fresh development environment..."

cd clients/python

if command -v uv &> /dev/null; then
    print_status "Using uv for Python environment..."
    uv venv --python 3.11 .venv
    source .venv/bin/activate
    
    # Install TensorZero client in development mode with agents optional dependency
    print_status "Installing TensorZero client with agents support..."
    uv sync --group dev --group agents
    uv run maturin develop --uv
else
    print_status "Using standard Python venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    
    # Install TensorZero client in development mode with agents optional dependency
    print_status "Installing TensorZero client with agents support..."
    pip install -e ".[agents]"
    pip install maturin
    maturin develop
fi

print_success "TensorZero Python client with Agents SDK integration ready"

# Test the installation
print_status "Testing installation..."

if python -c "import tensorzero; print('✅ TensorZero imported successfully')" 2>/dev/null; then
    print_success "TensorZero client working"
else
    print_error "Failed to import TensorZero client"
    exit 1
fi

if python -c "import tensorzero.agents; print('✅ TensorZero Agents integration imported successfully')" 2>/dev/null; then
    print_success "TensorZero Agents integration working"
else
    print_warning "TensorZero Agents integration not available (OpenAI Agents SDK may not be installed)"
fi

cd ../..

print_success "✅ Development environment fixed!"
echo ""
echo "📋 Next Steps:"
echo "=============="
echo ""
echo "1. Set your OpenAI API key:"
echo "   export OPENAI_API_KEY='sk-your-key-here'"
echo ""
echo "2. Test the integration:"
echo "   cd clients/python"
echo "   source .venv/bin/activate"
echo "   python -m pytest tests/test_agents_integration.py -v"
echo ""
echo "3. Or use the Makefile:"
echo "   make test-integration"
echo ""
echo "🎯 Environment ready! Use: pip install tensorzero[agents]" 