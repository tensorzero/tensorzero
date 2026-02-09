- The UI uses React Router 7, Tailwind, Node, and pnpm.
- UI routes (pages & API) are defined in `./app/routes.ts`.
- Prefer `logger` from `~/utils/logger` over `console.error`, `console.warn`, `console.log`, `console.debug`.
- Prefer using React Router's `useFetcher` over direct calls to `fetch`. [Reference](https://reactrouter.com/api/hooks/useFetcher)
- After modifying UI code, run from the `ui/` directory: `pnpm run format`, `pnpm run lint`, `pnpm run typecheck`. All commands must pass.

## Autopilot Feature (Internal Only)

> **Note for external contributors:** The Autopilot feature depends on a closed-source internal API. The tests in `e2e_tests/autopilot/` require access to the private `autopilot` repository and cannot be run by external contributors. These tests are run in CI via repository dispatch from the autopilot repo.

If you're an internal contributor with access to the autopilot repository:

### Setup

Set the `AUTOPILOT_REPO` environment variable to point to your local autopilot checkout:

```bash
export AUTOPILOT_REPO=/path/to/autopilot
```

### Running Autopilot Dependencies

```bash
docker compose --profile e2e up -d
```

Run from the `$AUTOPILOT_REPO` directory, or:

```bash
docker compose -f "$AUTOPILOT_REPO/docker-compose.yml" --profile e2e up -d
```

### Running the UI Dev Server with Autopilot

From the `ui/` directory:

```bash
TENSORZERO_UI_CONFIG_FILE="$AUTOPILOT_REPO/e2e_tests/fixtures/config/tensorzero.toml" \
TENSORZERO_GATEWAY_URL=http://localhost:3040 \
pnpm dev
```

### Running Autopilot E2E Tests

From the `ui/` directory:

```bash
TENSORZERO_UI_CONFIG_FILE="$AUTOPILOT_REPO/e2e_tests/fixtures/config/tensorzero.toml" \
TENSORZERO_GATEWAY_URL=http://localhost:3040 \
TENSORZERO_PLAYWRIGHT_INCLUDE_AUTOPILOT=1 \
pnpm exec playwright test e2e_tests/autopilot/
```

To run a specific test file:

```bash
TENSORZERO_UI_CONFIG_FILE="$AUTOPILOT_REPO/e2e_tests/fixtures/config/tensorzero.toml" \
TENSORZERO_GATEWAY_URL=http://localhost:3040 \
TENSORZERO_PLAYWRIGHT_INCLUDE_AUTOPILOT=1 \
pnpm exec playwright test e2e_tests/autopilot/autopilot.spec.ts
```
