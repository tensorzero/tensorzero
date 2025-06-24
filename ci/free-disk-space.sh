#!/bin/bash
set -euo pipefail

# Free up disk space for GitHub Actions runners
# This script aggressively removes large directories and packages not needed for most CI tasks
# Optimized for ClickHouse tests but can be used for other space-constrained jobs

echo "=== Disk usage before cleanup ==="
df -h
echo "=== Top disk usage by directory ==="
du -sh /usr/share/* 2>/dev/null | sort -hr | head -20 || true
du -sh /opt/* 2>/dev/null | sort -hr | head -20 || true

echo "=== Starting ultra-aggressive cleanup ==="

# Critical: Remove largest directories first (runs in parallel for speed)
echo "Removing critical large directories in parallel..."
(sudo rm -rf /usr/local/lib/android &)  # ~12GB Android SDK
(sudo rm -rf /usr/share/dotnet &)       # ~1.7GB .NET
(sudo rm -rf /opt/hostedtoolcache/CodeQL &)  # ~3GB CodeQL
(sudo rm -rf /opt/ghc &)                # ~1.6GB GHC
wait  # Wait for parallel deletions to complete

# Remove ALL hosted toolchains in parallel
echo "Removing ALL hosted toolchains in parallel..."
(sudo rm -rf /opt/hostedtoolcache/Python &)
(sudo rm -rf /opt/hostedtoolcache/node &)
(sudo rm -rf /opt/hostedtoolcache/go &)
(sudo rm -rf /opt/hostedtoolcache/Ruby &)
(sudo rm -rf /opt/hostedtoolcache/PyPy &)
(sudo rm -rf /opt/hostedtoolcache/Java_Temurin-Hotspot &)
wait

# Remove additional runtimes and tools
echo "Removing additional runtimes and cloud tools..."
(sudo rm -rf /usr/share/swift &)
(sudo rm -rf /opt/az &)  # Azure CLI
(sudo rm -rf /usr/share/miniconda &)
(sudo rm -rf /opt/pipx &)
(sudo rm -rf /opt/google &)  # Google Cloud SDK
(sudo rm -rf /usr/local/aws-cli &)  # AWS CLI
(sudo rm -rf /usr/local/share/powershell &)  # PowerShell
wait

# Ultra-aggressive Docker cleanup
echo "Ultra-aggressive Docker cleanup..."
docker system prune -af --volumes || true
sudo systemctl stop docker || true
sudo rm -rf /var/lib/docker/overlay2 || true
sudo rm -rf /var/lib/docker/containers || true
sudo rm -rf /var/lib/docker/image || true
sudo systemctl start docker || true

# APT and package cleanup
echo "Comprehensive package cleanup..."
sudo apt-get autoremove --purge -y || true
sudo apt-get autoclean || true
sudo apt-get clean || true
sudo rm -rf /var/lib/apt/lists/* || true
sudo rm -rf /var/cache/apt/archives/* || true

# Remove large log files and temporary files
echo "Cleaning logs and temporary files..."
sudo journalctl --vacuum-size=50M || true
sudo rm -rf /var/log/*.log* || true
sudo rm -rf /tmp/* || true
sudo rm -rf /var/tmp/* || true

# Remove man pages and documentation (CI tests don't need them)
echo "Removing documentation and man pages..."
sudo rm -rf /usr/share/man/* || true
sudo rm -rf /usr/share/doc/* || true
sudo rm -rf /usr/share/locale/* || true

# Remove Firefox and Chromium (CI tests don't need browsers)
echo "Removing browsers and GUI tools..."
sudo rm -rf /usr/bin/firefox* || true
sudo rm -rf /snap/chromium || true
sudo rm -rf /snap/firefox || true

echo "=== Final disk usage after ultra-aggressive cleanup ==="
df -h
echo "=== Verifying critical tools still work ==="
which cargo && echo "✓ Cargo available" || echo "⚠ Cargo not found"
which docker && echo "✓ Docker available" || echo "⚠ Docker not found"
which curl && echo "✓ Curl available" || echo "⚠ Curl not found"

echo "=== Disk cleanup completed successfully ==="