#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Generate and publish Allure Report to testing-dashboard GitHub Pages
# ------------------------------------------------------------------------------
# This script:
# 1. Installs Allure commandline tool
# 2. Converts JUnit XML to Allure results
# 3. Fetches historical data from testing-dashboard gh-pages
# 4. Generates HTML report with history
# 5. Pushes report to testing-dashboard gh-pages branch
#
# Required environment variables:
# - CI_BUILD_NUMBER: Build number for commit message (e.g., GitHub run_number)
# - CI_COMMIT_SHA: Commit SHA for commit message
# - TESTING_DASHBOARD_DEPLOY_KEY: SSH private key for pushing to testing-dashboard
# ------------------------------------------------------------------------------

JUNIT_XML_PATH="${1:-target/nextest/e2e/junit.xml}"
DASHBOARD_REPO_SSH="git@github.com:tensorzero/testing-dashboard.git"
DASHBOARD_URL="https://tensorzero.github.io/testing-dashboard/"

if [ ! -f "$JUNIT_XML_PATH" ]; then
    echo "Warning: JUnit XML file not found at $JUNIT_XML_PATH"
    echo "Skipping Allure Report generation"
    exit 0
fi

echo "Generating Allure Report from $JUNIT_XML_PATH..."

# ------------------------------------------------------------------------------
# Check for Java (required by Allure)
# ------------------------------------------------------------------------------
if ! command -v java &> /dev/null; then
    echo "Java is not installed. Installing OpenJDK..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y default-jre
    elif command -v yum &> /dev/null; then
        sudo yum install -y java-11-openjdk
    else
        echo "Error: Unable to install Java. Please install Java manually."
        exit 1
    fi
fi

java -version

# ------------------------------------------------------------------------------
# Install Allure commandline tool
# ------------------------------------------------------------------------------
ALLURE_VERSION="2.32.0"
ALLURE_TGZ="allure-${ALLURE_VERSION}.tgz"
ALLURE_DIR="/tmp/allure-${ALLURE_VERSION}"

echo "Installing Allure ${ALLURE_VERSION}..."
curl -o "$ALLURE_TGZ" -L "https://github.com/allure-framework/allure2/releases/download/${ALLURE_VERSION}/${ALLURE_TGZ}"
tar -zxf "$ALLURE_TGZ" -C /tmp/
export PATH="${ALLURE_DIR}/bin:$PATH"

# Verify installation
allure --version

# ------------------------------------------------------------------------------
# Prepare Allure results
# ------------------------------------------------------------------------------
echo "Preparing Allure results..."
mkdir -p allure-results
cp "$JUNIT_XML_PATH" allure-results/

# ------------------------------------------------------------------------------
# Set up SSH authentication
# ------------------------------------------------------------------------------
if [ -z "$TESTING_DASHBOARD_DEPLOY_KEY" ]; then
    echo "Error: TESTING_DASHBOARD_DEPLOY_KEY environment variable is not set"
    exit 1
fi

echo "Setting up SSH authentication..."
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Write SSH key to file
echo "$TESTING_DASHBOARD_DEPLOY_KEY" > ~/.ssh/testing_dashboard_deploy_key
chmod 600 ~/.ssh/testing_dashboard_deploy_key

# Configure SSH to use the deploy key for github.com
cat >> ~/.ssh/config <<EOF
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/testing_dashboard_deploy_key
    StrictHostKeyChecking no
EOF
chmod 600 ~/.ssh/config

# ------------------------------------------------------------------------------
# Configure git
# ------------------------------------------------------------------------------
echo "Configuring git..."
git config --global user.email "ci@tensorzero.com"
git config --global user.name "TensorZero CI"

# ------------------------------------------------------------------------------
# Retry logic for fetch, build, and push (handles concurrent PR pushes)
# ------------------------------------------------------------------------------
MAX_ATTEMPTS=3
ATTEMPT=1

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "=== Attempt $ATTEMPT of $MAX_ATTEMPTS ==="

    # Clean up from previous attempt if needed
    if [ $ATTEMPT -gt 1 ]; then
        echo "Cleaning up from previous attempt..."
        rm -rf gh-pages-repo allure-history allure-report

        # Add sleep with jitter to avoid collision with other concurrent runs
        # Base delay increases with each attempt: 5s, 10s, 15s
        BASE_DELAY=$((5 * $ATTEMPT))
        # Add random jitter between 0-5 seconds
        JITTER=$((RANDOM % 6))
        TOTAL_DELAY=$((BASE_DELAY + JITTER))
        echo "Waiting ${TOTAL_DELAY} seconds before retry..."
        sleep $TOTAL_DELAY
    fi

    # ------------------------------------------------------------------------------
    # Fetch historical data from testing-dashboard
    # ------------------------------------------------------------------------------
    echo "Fetching historical data from testing-dashboard..."
    if ! git clone --depth 1 --branch gh-pages "$DASHBOARD_REPO_SSH" gh-pages-repo; then
        echo "Failed to clone repository"
        ATTEMPT=$((ATTEMPT + 1))
        continue
    fi

    if [ -d "gh-pages-repo/allure-history" ]; then
        echo "Found existing Allure history"
        cp -r gh-pages-repo/allure-history allure-history
    else
        echo "No existing Allure history found, starting fresh"
        mkdir -p allure-history
    fi

    # ------------------------------------------------------------------------------
    # Generate Allure HTML report
    # ------------------------------------------------------------------------------
    echo "Generating Allure HTML report..."
    # Set ALLURE_RESULTS_LIMIT to keep more history (default is 20)
    export ALLURE_RESULTS_LIMIT=99999
    if ! allure generate --clean allure-results -o allure-report --history-dir allure-history; then
        echo "Failed to generate Allure report"
        ATTEMPT=$((ATTEMPT + 1))
        continue
    fi

    # Copy history for next run
    echo "Copying history for next run..."
    mkdir -p allure-report/allure-history
    cp -r allure-report/history/* allure-report/allure-history/ 2>/dev/null || true

    # ------------------------------------------------------------------------------
    # Update gh-pages branch in testing-dashboard
    # ------------------------------------------------------------------------------
    echo "Updating gh-pages branch..."
    cd gh-pages-repo

    # Update remote to use SSH
    git remote set-url origin "$DASHBOARD_REPO_SSH"

    # Remove old content and copy new report
    git rm -rf . || true
    cp -r ../allure-report/* .

    # ------------------------------------------------------------------------------
    # Commit and push to GitHub
    # ------------------------------------------------------------------------------
    echo "Committing and pushing report..."
    git add .

    # Create commit message using CI environment variables
    COMMIT_MSG="Update Allure Report - Build #${CI_BUILD_NUMBER} - ${CI_COMMIT_SHA:0:7}"
    if ! git commit -m "$COMMIT_MSG"; then
        echo "No changes to commit"
        cd ..
        break
    fi

    # Push using SSH authentication (no force, will fail if branch has moved)
    echo "Pushing to testing-dashboard repository..."
    if git push origin gh-pages; then
        echo "✓ Successfully pushed to testing-dashboard"
        cd ..
        break
    else
        echo "✗ Push failed, likely due to concurrent update"
        cd ..
        ATTEMPT=$((ATTEMPT + 1))

        if [ $ATTEMPT -le $MAX_ATTEMPTS ]; then
            echo "Will retry..."
        else
            echo "Max attempts reached, giving up"
            exit 1
        fi
    fi
done

if [ $ATTEMPT -gt $MAX_ATTEMPTS ]; then
    echo "Error: Failed to push report after $MAX_ATTEMPTS attempts"
    exit 1
fi

echo "✓ Allure Report successfully published!"
echo "View at: $DASHBOARD_URL"
