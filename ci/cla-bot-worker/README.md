# CLA Bot Worker

Cloudflare Worker that receives GitHub webhook events and enforces the TensorZero Contributor License Agreement on pull requests across every repo in the `tensorzero` org.

**This Worker is deployed manually. Changes here are not automatically deployed.**

## Behavior

- On `pull_request` (`opened`, `reopened`, `synchronize`): collects every distinct GitHub user who authored or committed any commit in the PR (plus the PR opener), drops `[bot]` accounts and allowlisted users, and compares the rest against `ci/cla-signatures.json` on the **target repo's** `cla-signatures` branch.
- On `issue_comment` (`created`) on a PR:
  - Body equals `recheck` → re-evaluate.
  - Body equals the canonical sign phrase → record a signature on the PR's repo, then re-evaluate.
- Each evaluation:
  - Posts a Check Run named `cla` on the PR head SHA. Conclusion is `success` (everyone signed) or `action_required` (someone hasn't).
  - Upserts a single sticky bot comment on the PR (identified by an HTML marker) listing who still needs to sign and the canonical phrase to copy-paste. If everyone was already signed when the PR was opened, the bot stays silent — only the green Check Run shows up.

### Multi-repo / org-wide

The bot acts on whichever repo a webhook arrives from, as long as the repo's owner matches `GITHUB_ORG`. To extend coverage, just install the GitHub App on more repos (or set the install scope to "All repositories"); no worker change is needed.

Each repo gets **its own** `cla-signatures` branch + `ci/cla-signatures.json`. The bot lazily creates the branch as an orphan commit on first need (no main-branch history pollution).

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
  - Contents: **Read & write** (commit signatures to each repo's `cla-signatures` branch; create the branch on first use)
  - Issues: **Read & write** (post and edit the sticky PR comment)
  - Pull requests: **Read & write** (list commits / metadata; required by GitHub for posting comments on PR conversations even though the underlying API is `issues.createComment`)
  - Checks: **Read & write** (post the `cla` Check Run)
  - Metadata: **Read**
- **Subscribe to events:** Pull request, Issue comment, Merge queue entry (`merge_group`)
- Install on the `tensorzero` organization with scope **All repositories** (covers current + future repos automatically).

If you change permissions on an existing app, the org owner must re-accept them on the installation page before they take effect. The installation ID does not change; no secret update needed.

## Per-repo branch protection (manual, optional)

For each repo where you want the CLA check to actually block merge, add the Check Run `cla` from `TensorZero CLA Bot` as a required status check on the protected branch (e.g. `main`). Repos without this rule still see the bot's check + comment, but merge isn't gated.
