#!/bin/bash
# Clippy Target and Feature Isolator
# This script helps identify which target or feature combination is causing a lint issue

# Define colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Clippy Target and Feature Isolator${NC}"
echo "This script will help identify which target or feature is causing lint warnings"
echo ""

# Get all available targets
echo -e "${YELLOW}Step 1: Collecting all targets...${NC}"
TARGETS=("--lib" "--bins" "--tests" "--examples" "--benches")

# Get all available features
echo -e "${YELLOW}Step 2: Collecting all features...${NC}"
FEATURES=$(cargo metadata --format-version=1 | jq -r '.packages[] | select(.name=="'$(basename $(pwd))'") | .features | keys[]')
FEATURE_LIST=($FEATURES)
echo "Found features: ${FEATURE_LIST[@]}"

# Add no-default-features and all-features to the testing mix
FEATURE_OPTIONS=("--no-default-features" "" "--all-features")
for feature in "${FEATURE_LIST[@]}"; do
  FEATURE_OPTIONS+=("--no-default-features --features $feature")
  FEATURE_OPTIONS+=("--features $feature")
done

# Function to run clippy and check for warnings
check_clippy() {
  local target=$1
  local feature_option=$2

  echo -e "\n${YELLOW}Testing:${NC} cargo clippy $target $feature_option -- -D warnings"

  # Run clippy and capture output and exit code
  OUTPUT=$(cargo clippy $target $feature_option -- -D warnings 2>&1)
  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ No warnings detected${NC}"
    return 0
  else
    echo -e "${RED}✗ Warnings/errors detected${NC}"
    echo "$OUTPUT" | grep -A 1 "warning:" | head -n 10
    # If there are more than 5 warnings, indicate that
    WARN_COUNT=$(echo "$OUTPUT" | grep -c "warning:")
    if [ $WARN_COUNT -gt 5 ]; then
      echo -e "...and $(($WARN_COUNT - 5)) more warnings"
    fi
    return 1
  fi
}

# First check all targets with all features (baseline)
echo -e "\n${YELLOW}Step 3: Baseline check with all targets and features${NC}"
check_clippy "--all-targets" "--all-features"
BASELINE_RESULT=$?

if [ $BASELINE_RESULT -eq 0 ]; then
  echo -e "\n${GREEN}No issues found with all targets and all features.${NC}"
  echo "This might mean the issue only appears in specific combinations that aren't covered by --all-targets or --all-features."
  echo "Continuing with detailed investigation..."
fi

# Test each target individually with all features
echo -e "\n${YELLOW}Step 4: Testing each target with all features${NC}"
PROBLEMATIC_TARGETS=()

for target in "${TARGETS[@]}"; do
  check_clippy "$target" "--all-features"
  if [ $? -ne 0 ]; then
    PROBLEMATIC_TARGETS+=("$target")
    echo -e "${RED}Found issue with target: $target${NC}"
  fi
done

# Test all targets with each feature option
echo -e "\n${YELLOW}Step 5: Testing all targets with each feature option${NC}"
PROBLEMATIC_FEATURES=()

for feature_option in "${FEATURE_OPTIONS[@]}"; do
  check_clippy "--all-targets" "$feature_option"
  if [ $? -ne 0 ]; then
    PROBLEMATIC_FEATURES+=("$feature_option")
    echo -e "${RED}Found issue with feature option: $feature_option${NC}"
  fi
done

# If we've identified problematic targets and features, do targeted testing
if [ ${#PROBLEMATIC_TARGETS[@]} -gt 0 ] && [ ${#PROBLEMATIC_FEATURES[@]} -gt 0 ]; then
  echo -e "\n${YELLOW}Step 6: Targeted testing of problematic combinations${NC}"

  for target in "${PROBLEMATIC_TARGETS[@]}"; do
    for feature_option in "${PROBLEMATIC_FEATURES[@]}"; do
      check_clippy "$target" "$feature_option"
      if [ $? -ne 0 ]; then
        echo -e "\n${RED}FOUND PROBLEMATIC COMBINATION:${NC}"
        echo -e "${RED}cargo clippy $target $feature_option -- -D warnings${NC}"
      fi
    done
  done
fi

# Summary
echo -e "\n${YELLOW}=== SUMMARY ===${NC}"
if [ ${#PROBLEMATIC_TARGETS[@]} -eq 0 ]; then
  echo "No specific problematic targets identified."
else
  echo -e "${RED}Problematic targets:${NC}"
  printf "  %s\n" "${PROBLEMATIC_TARGETS[@]}"
fi

if [ ${#PROBLEMATIC_FEATURES[@]} -eq 0 ]; then
  echo "No specific problematic feature options identified."
else
  echo -e "${RED}Problematic feature options:${NC}"
  printf "  %s\n" "${PROBLEMATIC_FEATURES[@]}"
fi

echo -e "\n${YELLOW}To check a specific combination, run:${NC}"
echo "cargo clippy [TARGET] [FEATURE_OPTION] -- -D warnings"
echo "For example: cargo clippy --lib --features some-feature -- -D warnings"
