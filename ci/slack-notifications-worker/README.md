# Slack Notifications Worker

Cloudflare Worker that receives GitHub webhook events and posts Slack notifications for external activity.

**This Worker is deployed manually. Changes here are not automatically deployed.**

## Deploy

```bash
npx wrangler deploy
```

## Secrets

Set via `wrangler secret put <NAME>`:

- `GITHUB_WEBHOOK_SECRET` — webhook secret configured in the GitHub App
- `SLACK_BOT_TOKEN` — Slack Bot User OAuth Token (`xoxb-...`)
- `GITHUB_APP_ID` — GitHub App ID (from app settings page)
- `GITHUB_APP_PRIVATE_KEY` — GitHub App private key (full PEM contents)
- `GITHUB_INSTALLATION_ID` — installation ID (from URL after installing: `/settings/installations/<ID>`)

## GitHub App Setup

- **Webhook URL:** Worker URL
- **Webhook Secret:** same as `GITHUB_WEBHOOK_SECRET`
- **Events:** Issues, Issue comments, Pull requests, Pull request reviews, Discussions, Discussion comments
