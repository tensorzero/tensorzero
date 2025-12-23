#! /bin/bash
# This script is a helper to run the TensorZero UI dev server pointed at the e2e test gateway and database.
# This may be helpful for debugging e2e tests or additional validation of the UI.
TENSORZERO_UI_CONFIG_PATH="../tensorzero-core/tests/e2e/config/tensorzero.*.toml" pnpm run dev
