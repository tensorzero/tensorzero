# TensorZero Bot Worker

Cloudflare Worker that receives GitHub webhook events and runs PR-housekeeping automation across every repo in the `tensorzero` org. Replaces two workflows that used `pull_request_target`:

- `.github/workflows/force-merge-queue.yml` — when the `force-add-to-merge-queue` label is added to a PR, post a `check-all-general-jobs-passed` commit status with state=success and rerun any failed `general.yml` workflow run for the head SHA.
- `.github/workflows/label-merge-conflicts.yml` — keep the `has-merge-conflicts` label in sync with each PR's `mergeable` state.

**This Worker is deployed manually. Changes here are not automatically deployed.**

## Behavior

### Force-merge-queue

- On `pull_request.labeled` with name = `force-add-to-merge-queue`:
  - Find any `check-all-general-jobs-passed` Check Run with conclusion `failure` on the PR head SHA, parse `details_url` to extract the workflow `run_id`, and call `actions.reRunWorkflow` (best-effort).
  - Post a `check-all-general-jobs-passed` commit status with state `success` on the PR head SHA.
- On `pull_request.unlabeled`: no-op (GitHub doesn't allow removing commit statuses; pushing a new commit will produce a fresh general.yml status that supersedes the bot's).
- The real merge-queue check still runs once the PR enters the queue, so this is always safe.

### Label-merge-conflicts

- On `pull_request.opened`, `pull_request.reopened`, `pull_request.synchronize`: re-evaluate that PR's `mergeable` state and add/remove the `has-merge-conflicts` label.
- On `push` to any branch: list open PRs whose `base` is the pushed ref, re-evaluate each in parallel (cap of 5 at a time).
- `mergeable` is computed lazily by GitHub, so the worker polls `pulls.get` up to 5 times with backoff (~10s total) until the value is non-null. If still `null`, the worker gives up; the next webhook event re-evaluates.
- The `has-merge-conflicts` label is auto-created on first use per repo.

## Deploy

```bash
npx wrangler deploy
```

## Secrets

Set via `wrangler secret put <NAME>`:

- `GITHUB_APP_ID`: GitHub App ID (from app settings page).
- `GITHUB_APP_PRIVATE_KEY`: GitHub App private key (must be PKCS#8 format; convert with `openssl pkcs8 -topk8 -inform PEM -outform PEM -nocrypt -in key.pem -out key-pkcs8.pem`).
- `GITHUB_INSTALLATION_ID`: installation ID (`gh api /orgs/tensorzero/installations --jq '.installations[] | select(.app_slug=="tensorzero-bot") | .id'`).
- `GITHUB_WEBHOOK_SECRET`: webhook secret configured in the GitHub App.

## GitHub App Setup

- **Name:** `TensorZero Bot`
- **Webhook URL:** Worker URL (`https://tensorzero-bot.tensorzero.workers.dev`)
- **Webhook Secret:** same as `GITHUB_WEBHOOK_SECRET`
- **Repository permissions:**
  - Pull requests: **Read & write** (list/get PRs, add/remove labels via the issues API)
  - Issues: **Read & write** (labels API; create the `has-merge-conflicts` label on first use)
  - Commit statuses: **Read & write** (post `check-all-general-jobs-passed` status)
  - Actions: **Read & write** (rerun failed `general.yml` runs)
  - Checks: **Read** (find failed runs by name)
  - Contents: **Read** (mandatory companion of pulls)
  - Metadata: **Read**
- **Subscribe to events:** Pull request, Push
- Install on the `tensorzero` organization with scope **All repositories** (covers current + future repos automatically).

If you change permissions on an existing app, the org owner must re-accept them on the installation page before they take effect. The installation ID does not change; no secret update needed.
