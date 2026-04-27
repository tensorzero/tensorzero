# CLA Bot Worker

Cloudflare Worker that receives GitHub webhook events and enforces the TensorZero Contributor License Agreement on pull requests.

**This Worker is deployed manually. Changes here are not automatically deployed.**

## Behavior

- On `pull_request` (`opened`, `reopened`, `synchronize`): collects every distinct GitHub user who authored or committed any commit in the PR (plus the PR opener), drops allowlisted entries and `[bot]` accounts, and compares against `ci/cla-signatures.json` on the `cla-signatures` branch.
- On `issue_comment` (`created`) on a PR:
  - Body equals `recheck` → re-evaluate.
  - Body equals the canonical sign phrase → record a signature, then re-evaluate.
- Each evaluation:
  - Posts a Check Run named `cla` on the PR head SHA. Conclusion is `success` (everyone signed) or `action_required` (someone hasn't).
  - Upserts a single sticky bot comment on the PR (identified by an HTML marker) listing who still needs to sign and the canonical phrase to copy-paste.

## Deploy

```bash
npx wrangler deploy
```

## Secrets

Set via `wrangler secret put <NAME>`:

- `GITHUB_APP_ID`: GitHub App ID (from app settings page).
- `GITHUB_APP_PRIVATE_KEY`: GitHub App private key (must be PKCS#8 format; convert with `openssl pkcs8 -topk8 -inform PEM -outform PEM -nocrypt -in key.pem -out key-pkcs8.pem`).
- `GITHUB_INSTALLATION_ID`: installation ID (run `gh api /orgs/tensorzero/installations --jq '.installations[] | "\(.app_slug) \(.id)"'`).
- `GITHUB_WEBHOOK_SECRET`: webhook secret configured in the GitHub App.

## GitHub App Setup

- **Name:** `TensorZero CLA Bot`
- **Webhook URL:** Worker URL (`https://tensorzero-cla-bot.tensorzero.workers.dev`)
- **Webhook Secret:** same as `GITHUB_WEBHOOK_SECRET`
- **Repository permissions:**
  - Contents: **Read & write** (commit signatures to the `cla-signatures` branch)
  - Issues: **Read & write** (post and edit the sticky PR comment)
  - Pull requests: **Read & write** (list commits / metadata; required by GitHub for posting comments on PR conversations even though the underlying API is `issues.createComment`)
  - Checks: **Read & write** (post the `cla` Check Run)
  - Metadata: **Read**
- **Subscribe to events:** Pull request, Issue comment
- Install on the `tensorzero` organization, scoped to `tensorzero/tensorzero`.

## Post-deploy: branch protection

After the Worker is live and posting Check Runs, update branch protection on `main`:

1. Remove the existing required commit-status check named `cla` (posted by `contributor-assistant/github-action`).
2. Add the new Check Run `cla` (posted by `TensorZero CLA Bot`) as a required status check.
3. Once the new check is required and passing on a real PR, delete `.github/workflows/cla.yml`.
