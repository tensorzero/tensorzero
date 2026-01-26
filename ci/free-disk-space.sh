#!/bin/bash
set -euo pipefail

# Free up disk space for GitHub Actions runners
# This script EXTREMELY aggressively removes large directories and packages not needed for CI tasks
# Optimized for ClickHouse tests - targeting 20GB+ space savings

# Minimum free space required (in GB) - exit early once achieved
REQUIRED_FREE_GB=25

# Check if we have enough free space and exit early if so
check_space_and_maybe_exit() {
    df -h
    local avail_kb
    avail_kb=$(df -k / | tail -1 | awk '{print $4}')
    local avail_gb=$((avail_kb / 1024 / 1024))
    if [ "$avail_gb" -ge "$REQUIRED_FREE_GB" ]; then
        echo "=== Sufficient free space achieved: ${avail_gb}GB >= ${REQUIRED_FREE_GB}GB ==="
        df -h
        echo "=== Early exit - disk cleanup completed successfully ==="
        exit 0
    fi
    echo "Current free space: ${avail_gb}GB (need ${REQUIRED_FREE_GB}GB)"
}

echo "=== Disk usage before cleanup ==="
check_space_and_maybe_exit
# echo "=== Detailed disk analysis ==="
# du -sh /usr/share/* 2>/dev/null | sort -hr | head -30 || true
# du -sh /opt/* 2>/dev/null | sort -hr | head -30 || true
# du -sh /usr/local/* 2>/dev/null | sort -hr | head -20 || true
# du -sh /var/* 2>/dev/null | sort -hr | head -20 || true

echo "=== Starting EXTREME cleanup - targeting 20GB+ savings ==="

# Phase 1: Remove largest directories first (parallel for speed)
echo "Phase 1: Removing massive directories in parallel..."
(sudo rm -rf /usr/local/lib/android &)  # ~12GB Android SDK
(sudo rm -rf /usr/share/dotnet &)       # ~1.7GB .NET
(sudo rm -rf /opt/hostedtoolcache/CodeQL &)  # ~3GB CodeQL
(sudo rm -rf /opt/ghc &)                # ~1.6GB GHC
(sudo rm -rf /usr/share/swift &)        # ~2.8GB Swift
wait
check_space_and_maybe_exit

# Phase 2: Remove ALL hosted toolchains (parallel)
echo "Phase 2: Removing ALL hosted toolchains..."
(sudo rm -rf /opt/hostedtoolcache/Python &)
(sudo rm -rf /opt/hostedtoolcache/node &)
(sudo rm -rf /opt/hostedtoolcache/go &)
(sudo rm -rf /opt/hostedtoolcache/Ruby &)
(sudo rm -rf /opt/hostedtoolcache/PyPy &)
(sudo rm -rf /opt/hostedtoolcache/Java_Temurin-Hotspot &)
(sudo rm -rf /opt/hostedtoolcache/* &)   # Remove any remaining toolchains
wait
check_space_and_maybe_exit

# Phase 3: Remove cloud tools and runtimes (parallel)
echo "Phase 3: Removing cloud tools and additional runtimes..."
(sudo rm -rf /opt/az &)                  # ~688MB Azure CLI
(sudo rm -rf /usr/share/miniconda &)     # ~698MB Miniconda
(sudo rm -rf /opt/pipx &)                # ~528MB pipx
(sudo rm -rf /opt/google &)              # ~366MB Google Cloud SDK
(sudo rm -rf /usr/local/share/powershell &)  # PowerShell
(sudo rm -rf /opt/microsoft &)           # ~772MB Microsoft tools
wait
check_space_and_maybe_exit

# Phase 4: EXTREME package removal - CI doesn't need most of these
echo "Phase 4: Removing unnecessary packages and development tools..."
(sudo rm -rf /usr/share/gradle-* &)      # ~146MB Gradle
(sudo rm -rf /usr/share/apache-maven-* &) # ~11MB Maven
(sudo rm -rf /usr/share/kotlinc &)       # ~91MB Kotlin
(sudo rm -rf /usr/share/ri &)            # ~56MB Ruby docs
(sudo rm -rf /usr/share/mecab &)         # ~52MB MeCab
(sudo rm -rf /usr/share/java &)          # ~44MB Java files
(sudo rm -rf /usr/share/vim &)           # ~42MB Vim
(sudo rm -rf /usr/share/fonts &)         # ~36MB Fonts
(sudo rm -rf /usr/share/icons &)         # ~47MB Icons
(sudo rm -rf /usr/share/python-babel-localedata &) # ~31MB Python locale
wait
check_space_and_maybe_exit

# Phase 5: Ultra-aggressive system cleanup
echo "Phase 5: EXTREME Docker and system cleanup..."
# Stop all containers and remove everything
docker kill $(docker ps -q) 2>/dev/null || true
docker rm $(docker ps -a -q) 2>/dev/null || true
docker rmi $(docker images -q) 2>/dev/null || true
docker system prune -af --volumes || true
sudo systemctl stop docker || true
sudo rm -rf /var/lib/docker/* || true
sudo rm -rf /var/lib/containerd/* || true
sudo systemctl start docker || true
check_space_and_maybe_exit

# Phase 6: Snap packages removal (they take significant space)
echo "Phase 6: Removing snap packages..."
sudo snap list 2>/dev/null | awk 'NR>1 {print $1}' | xargs -r sudo snap remove || true
sudo rm -rf /var/lib/snapd/snaps/* || true
sudo rm -rf /snap/* || true
check_space_and_maybe_exit

# Phase 7: Remove large system components not needed for CI
echo "Phase 7: Removing large system components..."
(sudo rm -rf /usr/lib/x86_64-linux-gnu/dri/* &)  # Graphics drivers
(sudo rm -rf /usr/lib/modules/*/kernel/drivers/gpu/* &)  # GPU drivers
(sudo rm -rf /usr/lib/firmware/* &)              # Hardware firmware
(sudo rm -rf /var/cache/fontconfig/* &)          # Font cache
(sudo rm -rf /usr/share/X11/* &)                 # X11 files
(sudo rm -rf /usr/share/pixmaps/* &)             # Pixmaps
(sudo rm -rf /usr/share/applications/* &)        # Desktop applications
wait
check_space_and_maybe_exit

# Phase 8: Aggressive APT cleanup with package removal
echo "Phase 8: Extreme APT cleanup and package removal..."
# Remove packages we definitely don't need for Rust/ClickHouse testing
sudo apt-get remove --purge -y \
  firefox* chromium* \
  thunderbird* \
  libreoffice* \
  gimp* \
  vlc* \
  imagemagick* \
  php* \
  apache2* \
  nginx* \
  mysql* \
  postgresql* \
  mongodb* \
  redis* \
  elasticsearch* \
  2>/dev/null || true

sudo apt-get autoremove --purge -y || true
sudo apt-get autoclean || true
sudo apt-get clean || true
sudo rm -rf /var/lib/apt/lists/* || true
sudo rm -rf /var/cache/apt/* || true
sudo rm -rf /var/lib/dpkg/info/*.list || true
check_space_and_maybe_exit

# Phase 9: Remove all documentation, manuals, and locales
echo "Phase 9: Removing ALL documentation and locales..."
sudo rm -rf /usr/share/man/* || true
sudo rm -rf /usr/share/doc/* || true
sudo rm -rf /usr/share/info/* || true
sudo rm -rf /usr/share/locale/* || true
sudo rm -rf /usr/share/i18n/locales/* || true
sudo rm -rf /usr/lib/locale/* || true
check_space_and_maybe_exit

# Phase 10: Clean logs, cache, and temporary files extremely aggressively
echo "Phase 10: Extreme cleanup of logs, cache, and temp files..."
sudo journalctl --vacuum-size=10M || true
sudo rm -rf /var/log/* || true
sudo rm -rf /tmp/* || true
sudo rm -rf /var/tmp/* || true
sudo rm -rf /root/.cache/* || true
sudo rm -rf /home/*/.cache/* || true
sudo rm -rf /var/cache/* || true
sudo rm -rf /usr/share/mime/* || true
check_space_and_maybe_exit

# Phase 11: Remove kernel modules and headers we don't need
echo "Phase 11: Removing unnecessary kernel components..."
sudo rm -rf /lib/modules/*/kernel/sound/* || true
sudo rm -rf /lib/modules/*/kernel/drivers/media/* || true
sudo rm -rf /lib/modules/*/kernel/drivers/staging/* || true
sudo rm -rf /usr/src/linux-headers-* || true

# echo "=== EXTREME cleanup completed - checking results ==="
# df -h
# echo "=== Verifying critical tools still work ==="
# which cargo && echo "✓ Cargo available" || echo "⚠ Cargo not found"
# which docker && echo "✓ Docker available" || echo "⚠ Docker not found"
# which curl && echo "✓ Curl available" || echo "⚠ Curl not found"
# which git && echo "✓ Git available" || echo "⚠ Git not found"
# which rustc && echo "✓ Rust available" || echo "⚠ Rust not found"

# echo "=== Remaining large directories (>100MB) ==="
# du -sh /usr/* 2>/dev/null | awk '$1 ~ /[0-9]+[GM]/ && $1+0 >= 100 {print}' | sort -hr | head -20 || true
# du -sh /opt/* 2>/dev/null | awk '$1 ~ /[0-9]+[GM]/ && $1+0 >= 100 {print}' | sort -hr | head -10 || true

echo "=== EXTREME disk cleanup completed successfully ==="
df -h
