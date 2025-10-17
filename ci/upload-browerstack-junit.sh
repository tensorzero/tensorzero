set -euxo pipefail

curl -u "aaronhill_foWj4v:$BROWSERSTACK_KEY" -vvv \
    -X POST \
    -F "data=@$BROWSERSTACK_JUNIT_FILE" \
    -F "projectName=TensorZero E2E tests" \
    -F "buildName=$BROWSERSTACK_BUILD_NAME" \
    -F "buildIdentifier=$BROWSERSTACK_RUN_ID" \
    -F "ci=$BROWSERSTACK_CI_URL" \
https://upload-automation.browserstack.com/upload

