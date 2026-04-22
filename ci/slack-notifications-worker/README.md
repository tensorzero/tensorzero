# Slack Notifications Worker

Cloudflare Worker that receives GitHub webhook events and posts Slack notifications for external activity.

**This Worker is deployed manually. Changes here are not automatically deployed.**

## Deploy

```bash
npx wrangler deploy
```

## Secrets

Set via `wrangler secret put <NAME>`:

- `GITHUB_APP_ID`: GitHub App ID (from app settings page)
- `GITHUB_APP_PRIVATE_KEY`: GitHub App private key (must be PKCS#8 format; convert with `openssl pkcs8 -topk8 -inform PEM -outform PEM -nocrypt -in key.pem -out key-pkcs8.pem`)
- `GITHUB_INSTALLATION_ID`: installation ID (run `gh api /orgs/tensorzero/installations --jq '.installations[] | "\(.app_slug) \(.id)"'`)
- `GITHUB_WEBHOOK_SECRET`: webhook secret configured in the GitHub App
- `SLACK_BOT_TOKEN`: Slack Bot User OAuth Token (`xoxb-...`)

## GitHub App Setup

- **Webhook URL:** Worker URL (`https://tensorzero-slack-notifications.tensorzero.workers.dev`)
- **Webhook Secret:** same as `GITHUB_WEBHOOK_SECRET`
- **Repository permissions:** Issues (read), Pull requests (read), Discussions (read)
- **Organization permissions:** Members (read)
- **Events:** Issues, Issue comments, Pull requests, Pull request reviews, Discussions, Discussion comments
