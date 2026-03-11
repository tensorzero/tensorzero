#!/usr/bin/env bash
# Validates that the local development environment has all required tools installed.
# See CONTRIBUTING.md for setup instructions.
set -euo pipefail

errors=0

check() {
    local name="$1"
    local cmd="$2"
    local install_hint="$3"

    if eval "$cmd" > /dev/null 2>&1; then
        printf "  %-20s ✓\n" "$name"
    else
        printf "  %-20s ✗  %s\n" "$name" "$install_hint"
        errors=$((errors + 1))
    fi
}

check_optional() {
    local name="$1"
    local cmd="$2"
    local hint="$3"

    if eval "$cmd" > /dev/null 2>&1; then
        printf "  %-20s ✓\n" "$name"
    else
        printf "  %-20s ○  (optional) %s\n" "$name" "$hint"
    fi
}

echo "Checking development environment..."
echo ""

echo "Required tools:"
check "rustc"          "rustc --version"                      "https://www.rust-lang.org/tools/install"
check "cargo"          "cargo --version"                      "https://www.rust-lang.org/tools/install"
check "cargo-nextest"  "cargo nextest --version"              "https://nexte.st/docs/installation/pre-built-binaries/"
check "cargo-deny"     "cargo deny --version"                 "https://github.com/EmbarkStudios/cargo-deny"
check "docker"         "docker --version"                     "https://docs.docker.com/get-docker/"
check "uv"            "uv --version"                          "https://docs.astral.sh/uv/"
check "python3"       "python3 --version"                     "uv python install 3.9"
check "node"          "node --version"                        "https://nodejs.org/en"
check "pnpm"          "pnpm --version"                        "npm install -g pnpm@10.15.0"
check "pre-commit"    "pre-commit --version"                  "https://pre-commit.com/#install"

# mold is required on Linux (configured in .cargo/config.toml)
if [[ "$(uname -s)" == "Linux" ]]; then
    check "mold"      "mold --version"                        "https://github.com/rui314/mold#installation"
fi

echo ""
echo "Optional tools:"
check_optional "cargo-hack"   "cargo hack --version"          "https://github.com/DevinR528/cargo-hack"

if [[ "$(uname -s)" == "Darwin" ]]; then
    check_optional "mold"     "mold --version"                "brew install mold — faster linking on macOS"
fi

echo ""
if [[ $errors -gt 0 ]]; then
    echo "Found $errors missing required tool(s). See CONTRIBUTING.md for setup instructions."
    exit 1
else
    echo "All required tools are installed."
fi
