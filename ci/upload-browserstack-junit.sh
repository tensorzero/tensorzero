#!/bin/bash
set -euxo pipefail

# cd to the script's directory
cd "$(dirname "$0")"

python3 ./process_flaky_tests.py $BROWSERSTACK_JUNIT_FILE -o ./processed_junit.xml

curl -u "aaronhill_foWj4v:$BROWSERSTACK_KEY" -vvv \
    -X POST \
    -F "data=@./processed_junit.xml" \
    -F "projectName=TensorZero E2E tests" \
    -F "buildName=$BROWSERSTACK_BUILD_NAME" \
    -F "buildIdentifier=$BROWSERSTACK_RUN_ID" \
    -F "ci=$BROWSERSTACK_CI_URL" \
https://upload-automation.browserstack.com/upload

