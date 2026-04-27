// Cloudflare Worker that receives GitHub webhook events and enforces the
// TensorZero Contributor License Agreement on pull requests.
//
// Required secrets (set via `wrangler secret put`):
//   GITHUB_APP_ID          - GitHub App ID
//   GITHUB_APP_PRIVATE_KEY - GitHub App private key (PKCS#8 PEM)
//   GITHUB_INSTALLATION_ID - GitHub App installation ID for the tensorzero org
//   GITHUB_WEBHOOK_SECRET  - the webhook secret configured in the GitHub App
//
// Required vars (set in wrangler.toml):
//   GITHUB_ORG, GITHUB_REPO
//   CLA_BRANCH, CLA_SIGNATURES_PATH, CLA_DOC_URL
//   CHECK_NAME
//   SIGN_PHRASE, RECHECK_PHRASE
//   ALLOWLIST (comma-separated GitHub logins)

import { App } from "@octokit/app";
import { Octokit } from "@octokit/core";
import { paginateRest } from "@octokit/plugin-paginate-rest";
import { restEndpointMethods } from "@octokit/plugin-rest-endpoint-methods";

const MyOctokit = Octokit.plugin(paginateRest, restEndpointMethods);

const COMMENT_MARKER = "<!-- tz-cla-bot -->";

export default {
  async fetch(request, env) {
    if (request.method !== "POST") {
      return new Response("Method not allowed", { status: 405 });
    }

    const rawBody = await request.text();

    const signature = request.headers.get("X-Hub-Signature-256");
    if (!signature) {
      return new Response("Unauthorized", { status: 401 });
    }
    const app = createApp(env);
    const isValid = await app.webhooks.verify(rawBody, signature);
    if (!isValid) {
      return new Response("Unauthorized", { status: 401 });
    }

    const event = request.headers.get("X-GitHub-Event");
    const payload = JSON.parse(rawBody);

    const expectedRepo = `${env.GITHUB_ORG}/${env.GITHUB_REPO}`;
    if (payload.repository?.full_name !== expectedRepo) {
      return new Response("OK (skipped: wrong repo)", { status: 200 });
    }

    const octokit = await app.getInstallationOctokit(
      Number(env.GITHUB_INSTALLATION_ID),
    );

    if (event === "pull_request") {
      if (!["opened", "reopened", "synchronize"].includes(payload.action)) {
        return new Response("OK (skipped action)", { status: 200 });
      }
      await evaluatePr(octokit, payload.pull_request, env);
      return new Response("OK", { status: 200 });
    }

    if (event === "issue_comment") {
      if (payload.action !== "created") {
        return new Response("OK (skipped action)", { status: 200 });
      }
      if (!payload.issue?.pull_request) {
        return new Response("OK (skipped: not a PR)", { status: 200 });
      }
      const commentBody = (payload.comment.body || "").trim();
      const isSign = commentBody === env.SIGN_PHRASE;
      const isRecheck = commentBody === env.RECHECK_PHRASE;
      if (!isSign && !isRecheck) {
        return new Response("OK (skipped: not a CLA command)", { status: 200 });
      }

      const { data: pr } = await octokit.rest.pulls.get({
        owner: env.GITHUB_ORG,
        repo: env.GITHUB_REPO,
        pull_number: payload.issue.number,
      });

      if (isSign) {
        await recordSignature(octokit, payload.comment, pr, env);
      }
      await evaluatePr(octokit, pr, env);
      return new Response("OK", { status: 200 });
    }

    return new Response("OK (skipped event)", { status: 200 });
  },
};

function createApp(env) {
  return new App({
    appId: env.GITHUB_APP_ID,
    privateKey: env.GITHUB_APP_PRIVATE_KEY,
    webhooks: { secret: env.GITHUB_WEBHOOK_SECRET },
    Octokit: MyOctokit,
  });
}

// --- Evaluate a PR: post Check Run + sticky comment ---

async function evaluatePr(octokit, pr, env) {
  const allowlist = parseAllowlist(env.ALLOWLIST);

  const commits = await octokit.paginate(octokit.rest.pulls.listCommits, {
    owner: env.GITHUB_ORG,
    repo: env.GITHUB_REPO,
    pull_number: pr.number,
    per_page: 100,
  });

  const candidates = new Map();
  const addUser = (user) => {
    if (!user || !user.login || !user.id) return;
    // GitHub's web-flow committer represents browser-side commits (e.g.
    // squash merges). It's a synthetic identity, not a real contributor.
    if (user.login === "web-flow") return;
    candidates.set(user.id, { login: user.login, id: user.id });
  };

  addUser(pr.user);
  for (const c of commits) {
    addUser(c.author);
    addUser(c.committer);
  }

  const required = [...candidates.values()].filter(
    (u) => !shouldSkip(u.login, allowlist),
  );

  const signatures = await readSignatures(octokit, env);
  const signedIds = new Set(signatures.signedContributors.map((s) => s.id));
  const unsigned = required.filter((u) => !signedIds.has(u.id));

  const allSigned = unsigned.length === 0;
  await octokit.rest.checks.create({
    owner: env.GITHUB_ORG,
    repo: env.GITHUB_REPO,
    name: env.CHECK_NAME,
    head_sha: pr.head.sha,
    status: "completed",
    conclusion: allSigned ? "success" : "action_required",
    details_url: env.CLA_DOC_URL,
    output: {
      title: allSigned
        ? "All contributors have signed the CLA"
        : "CLA signature required",
      summary: buildCheckSummary(unsigned, env),
    },
  });

  await upsertStickyComment(octokit, pr, unsigned, env);
}

function buildCheckSummary(unsigned, env) {
  if (unsigned.length === 0) {
    return `All contributors to this pull request have signed the [Contributor License Agreement](${env.CLA_DOC_URL}).`;
  }
  const list = unsigned.map((u) => `- @${u.login}`).join("\n");
  return [
    `The following contributors still need to sign the [Contributor License Agreement](${env.CLA_DOC_URL}):`,
    "",
    list,
    "",
    "To sign, add the following comment to this pull request:",
    "",
    `> ${env.SIGN_PHRASE}`,
  ].join("\n");
}

async function upsertStickyComment(octokit, pr, unsigned, env) {
  const comments = await octokit.paginate(octokit.rest.issues.listComments, {
    owner: env.GITHUB_ORG,
    repo: env.GITHUB_REPO,
    issue_number: pr.number,
    per_page: 100,
  });
  const existing = comments.find((c) =>
    (c.body || "").includes(COMMENT_MARKER),
  );

  // If everyone was already signed when the PR was opened, stay silent —
  // the green Check Run is enough. Only edit the comment if we previously
  // posted one (i.e. someone went from unsigned to signed in this PR).
  if (unsigned.length === 0 && !existing) return;

  const body = renderStickyBody(unsigned, env);
  if (existing) {
    if ((existing.body || "") === body) return;
    await octokit.rest.issues.updateComment({
      owner: env.GITHUB_ORG,
      repo: env.GITHUB_REPO,
      comment_id: existing.id,
      body,
    });
    return;
  }

  await octokit.rest.issues.createComment({
    owner: env.GITHUB_ORG,
    repo: env.GITHUB_REPO,
    issue_number: pr.number,
    body,
  });
}

function renderStickyBody(unsigned, env) {
  if (unsigned.length === 0) {
    return [
      COMMENT_MARKER,
      "",
      `✅ All contributors to this pull request have signed the [TensorZero CLA](${env.CLA_DOC_URL}). Thank you!`,
    ].join("\n");
  }
  const list = unsigned.map((u) => `- [ ] @${u.login}`).join("\n");
  return [
    COMMENT_MARKER,
    "",
    `Thank you for your contribution! Before we can accept this pull request, we need every commit author to sign the [TensorZero Contributor License Agreement](${env.CLA_DOC_URL}).`,
    "",
    "**Pending signatures:**",
    "",
    list,
    "",
    "If that's you, please post a comment on this pull request with the following text:",
    "",
    `> ${env.SIGN_PHRASE}`,
    "",
    `If anything looks off, comment \`${env.RECHECK_PHRASE}\` and the bot will re-evaluate.`,
  ].join("\n");
}

// --- Record a signature: append to cla-signatures.json with retry ---

async function recordSignature(octokit, comment, pr, env) {
  const newEntry = {
    name: comment.user.login,
    id: comment.user.id,
    comment_id: comment.id,
    created_at: comment.created_at,
    repoId: pr.base.repo.id,
    pullRequestNo: pr.number,
  };

  const maxAttempts = 5;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const { ref, signatures } = await readSignaturesWithSha(octokit, env);

    if (signatures.signedContributors.some((s) => s.id === newEntry.id)) {
      return;
    }
    signatures.signedContributors.push(newEntry);

    try {
      await commitSignatures(octokit, env, ref, signatures, newEntry);
      return;
    } catch (err) {
      if (err.status === 422 && attempt < maxAttempts) {
        await sleep(100 + Math.floor(Math.random() * 200));
        continue;
      }
      throw err;
    }
  }
}

async function readSignatures(octokit, env) {
  const { signatures } = await readSignaturesWithSha(octokit, env);
  return signatures;
}

async function readSignaturesWithSha(octokit, env) {
  const { data: ref } = await octokit.rest.git.getRef({
    owner: env.GITHUB_ORG,
    repo: env.GITHUB_REPO,
    ref: `heads/${env.CLA_BRANCH}`,
  });

  const { data: file } = await octokit.rest.repos.getContent({
    owner: env.GITHUB_ORG,
    repo: env.GITHUB_REPO,
    path: env.CLA_SIGNATURES_PATH,
    ref: env.CLA_BRANCH,
  });

  const content = decodeBase64Utf8(file.content);
  const signatures = JSON.parse(content);
  return { ref, signatures };
}

async function commitSignatures(octokit, env, ref, signatures, newEntry) {
  const { data: parentCommit } = await octokit.rest.git.getCommit({
    owner: env.GITHUB_ORG,
    repo: env.GITHUB_REPO,
    commit_sha: ref.object.sha,
  });

  const newContent = JSON.stringify(signatures, null, 2) + "\n";
  const { data: blob } = await octokit.rest.git.createBlob({
    owner: env.GITHUB_ORG,
    repo: env.GITHUB_REPO,
    content: newContent,
    encoding: "utf-8",
  });

  const { data: tree } = await octokit.rest.git.createTree({
    owner: env.GITHUB_ORG,
    repo: env.GITHUB_REPO,
    base_tree: parentCommit.tree.sha,
    tree: [
      {
        path: env.CLA_SIGNATURES_PATH,
        mode: "100644",
        type: "blob",
        sha: blob.sha,
      },
    ],
  });

  const { data: commit } = await octokit.rest.git.createCommit({
    owner: env.GITHUB_ORG,
    repo: env.GITHUB_REPO,
    message: `Sign CLA: @${newEntry.name} (#${newEntry.pullRequestNo})`,
    tree: tree.sha,
    parents: [parentCommit.sha],
  });

  await octokit.rest.git.updateRef({
    owner: env.GITHUB_ORG,
    repo: env.GITHUB_REPO,
    ref: `heads/${env.CLA_BRANCH}`,
    sha: commit.sha,
    force: false,
  });
}

// --- Helpers ---

function shouldSkip(login, allowlist) {
  const lower = login.toLowerCase();
  if (lower.endsWith("[bot]")) return true;
  return allowlist.some((entry) => entry.toLowerCase() === lower);
}

function parseAllowlist(csv) {
  if (!csv) return [];
  return csv
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function decodeBase64Utf8(b64) {
  const cleaned = b64.replace(/\n/g, "");
  const bin = atob(cleaned);
  const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
  return new TextDecoder("utf-8").decode(bytes);
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
