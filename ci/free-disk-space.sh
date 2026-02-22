#!/bin/bash
set -euo pipefail

# Free up disk space on GitHub Actions Ubuntu runners.
# Based on https://github.com/jlumbroso/free-disk-space
# Runs cleanup phases in order (fastest first) and exits early
# once the target free-space threshold is reached.

TARGET_KB=$((25 * 1024 * 1024)) # 25 GB in KB

check_and_maybe_exit() {
    local avail_kb
    avail_kb=$(df -k / | tail -1 | awk '{print $4}')
    local avail_gb=$((avail_kb / 1024 / 1024))
    if [ "$avail_kb" -ge "$TARGET_KB" ]; then
        echo "=== Free space target reached: ${avail_gb} GB available (>= 25 GB) ==="
        df -h /
        exit 0
    fi
    echo "--- Free space: ${avail_gb} GB (target: 25 GB) ---"
}

echo "=== Disk space before cleanup ==="
df -h /
check_and_maybe_exit

# Phase 1: Android SDK (~12 GB)
echo "Phase 1: Removing Android SDK..."
sudo rm -rf /usr/local/lib/android || true
check_and_maybe_exit

# Phase 2: .NET runtime (~1.7 GB)
echo "Phase 2: Removing .NET runtime..."
sudo rm -rf /usr/share/dotnet || true
check_and_maybe_exit

# Phase 3: Haskell runtime (~1.6 GB)
echo "Phase 3: Removing Haskell runtime..."
sudo rm -rf /opt/ghc /usr/local/.ghcup || true
check_and_maybe_exit

# Phase 4: Swap storage
echo "Phase 4: Reclaiming swap storage..."
sudo swapoff -a || true
sudo rm -f /mnt/swapfile || true
check_and_maybe_exit

# Phase 5: Docker images
echo "Phase 5: Pruning Docker images..."
sudo docker image prune --all --force || true
check_and_maybe_exit

# Phase 6: Large packages (slowest)
# Each removal is separate so one failure doesn't block the rest.
echo "Phase 6: Removing large unnecessary packages..."
sudo apt-get remove -y '^aspnetcore-.*' --fix-missing || true
sudo apt-get remove -y '^dotnet-.*' --fix-missing || true
sudo apt-get remove -y '^llvm-.*' --fix-missing || true
sudo apt-get remove -y 'php.*' --fix-missing || true
sudo apt-get remove -y '^mongodb-.*' --fix-missing || true
sudo apt-get remove -y '^mysql-.*' --fix-missing || true
sudo apt-get remove -y azure-cli google-chrome-stable firefox powershell mono-devel libgl1-mesa-dri --fix-missing || true
sudo apt-get remove -y google-cloud-sdk --fix-missing || true
sudo apt-get remove -y google-cloud-cli --fix-missing || true
sudo apt-get autoremove -y || true
sudo apt-get clean || true
check_and_maybe_exit

echo "=== Disk cleanup completed ==="
df -h /
