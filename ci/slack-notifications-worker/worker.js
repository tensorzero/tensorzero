// Cloudflare Worker that receives GitHub webhook events and posts
// Slack notifications for external (non-org-member) activity.
//
// Required secrets (set via `wrangler secret put`):
//   GITHUB_WEBHOOK_SECRET  - the webhook secret configured in the GitHub App
//   SLACK_BOT_TOKEN        - Slack Bot User OAuth Token (xoxb-...)
//   GITHUB_APP_ID          - GitHub App ID
//   GITHUB_APP_PRIVATE_KEY - GitHub App private key (PEM)
//   GITHUB_INSTALLATION_ID - GitHub App installation ID for the tensorzero org
//
// Required vars (set in wrangler.toml):
//   SLACK_CHANNEL          - Slack channel ID to post to
//   GITHUB_ORG             - GitHub organization name

import { App } from "@octokit/app";

export default {
  async fetch(request, env) {
    if (request.method !== "POST") {
      return new Response("Method not allowed", { status: 405 });
    }

    const body = await request.text();

    // Verify webhook signature using Octokit
    const app = createApp(env);
    const isValid = await app.webhooks.verify(body, request.headers.get("X-Hub-Signature-256") || "");
    if (!isValid) {
      return new Response("Unauthorized", { status: 401 });
    }

    const event = request.headers.get("X-GitHub-Event");
    const payload = JSON.parse(body);

    const notification = buildNotification(event, payload, env);
    if (!notification) {
      return new Response("OK (skipped)", { status: 200 });
    }

    if (await shouldSkip(notification.actor, env)) {
      return new Response("OK (skipped: org member or bot)", { status: 200 });
    }

    await postToSlack(notification, env);
    return new Response("OK", { status: 200 });
  },
};

function createApp(env) {
  return new App({
    appId: env.GITHUB_APP_ID,
    privateKey: env.GITHUB_APP_PRIVATE_KEY,
    webhooks: { secret: env.GITHUB_WEBHOOK_SECRET },
  });
}

// --- Build notification from event ---

function buildNotification(event, payload, env) {
  const channel = env.SLACK_CHANNEL;

  switch (event) {
    case "issues":
      if (payload.action !== "opened") return null;
      return {
        actor: payload.issue.user.login, channel,
        text: `New issue opened by ${payload.issue.user.login}`,
        blocks: formatBlock("New issue opened", payload.issue.title, payload.issue.html_url, payload.issue.user.login),
      };

    case "issue_comment":
      if (payload.action !== "created") return null;
      return {
        actor: payload.comment.user.login, channel,
        text: `New comment on issue by ${payload.comment.user.login}`,
        blocks: formatBlock("New comment on issue", payload.issue.title, payload.comment.html_url, payload.comment.user.login),
      };

    case "pull_request":
      if (payload.action !== "opened") return null;
      return {
        actor: payload.pull_request.user.login, channel,
        text: `New pull request opened by ${payload.pull_request.user.login}`,
        blocks: formatBlock("New pull request opened", payload.pull_request.title, payload.pull_request.html_url, payload.pull_request.user.login),
      };

    case "pull_request_review":
      if (payload.action !== "submitted") return null;
      return {
        actor: payload.review.user.login, channel,
        text: `Pull request review submitted by ${payload.review.user.login}`,
        blocks: formatBlock("Pull request review submitted", payload.pull_request.title, payload.pull_request.html_url, payload.review.user.login),
      };

    case "discussion":
      if (payload.action !== "created") return null;
      return {
        actor: payload.discussion.user.login, channel,
        text: `New discussion created by ${payload.discussion.user.login}`,
        blocks: formatBlock("New discussion created", payload.discussion.title, payload.discussion.html_url, payload.discussion.user.login),
      };

    case "discussion_comment":
      if (payload.action !== "created") return null;
      return {
        actor: payload.comment.user.login, channel,
        text: `New comment on discussion by ${payload.comment.user.login}`,
        blocks: formatBlock("New comment on discussion", payload.discussion.title, payload.comment.html_url, payload.comment.user.login),
      };

    default:
      return null;
  }
}

function formatBlock(heading, title, url, actor) {
  const actorUrl = `https://github.com/${actor}`;
  return [{
    type: "section",
    text: {
      type: "mrkdwn",
      text: `*${heading}*\n\n*Title:* <${url}|${title}>\n*Author:* <${actorUrl}|${actor}>`,
    },
  }];
}

// --- Skip logic ---

async function shouldSkip(actor, env) {
  if (actor.toLowerCase().endsWith("[bot]")) return true;

  const app = createApp(env);
  const octokit = await app.getInstallationOctokit(Number(env.GITHUB_INSTALLATION_ID));
  try {
    await octokit.rest.orgs.checkMembershipForUser({
      org: env.GITHUB_ORG,
      username: actor,
    });
    return true; // 204 = is a member
  } catch {
    return false; // 404 = not a member
  }
}

// --- Post to Slack ---

async function postToSlack(notification, env) {
  await fetch("https://slack.com/api/chat.postMessage", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.SLACK_BOT_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      channel: notification.channel,
      text: notification.text,
      blocks: notification.blocks,
    }),
  });
}
