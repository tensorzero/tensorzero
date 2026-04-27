// Cloudflare Worker that receives GitHub webhook events and enforces the
// TensorZero Contributor License Agreement on pull requests across every
// repo in the tensorzero org.
//
// Required secrets (set via `wrangler secret put`):
//   GITHUB_APP_ID          - GitHub App ID
//   GITHUB_APP_PRIVATE_KEY - GitHub App private key (PKCS#8 PEM)
//   GITHUB_INSTALLATION_ID - GitHub App installation ID for the tensorzero org
//   GITHUB_WEBHOOK_SECRET  - the webhook secret configured in the GitHub App
//
// Required vars (set in wrangler.toml):
//   GITHUB_ORG
//   CLA_BRANCH, CLA_SIGNATURES_PATH, CLA_DOC_URL
//   CHECK_NAME
//   SIGN_PHRASE, RECHECK_PHRASE
//   ALLOWLIST (comma-separated GitHub logins)

import { App } from "@octokit/app";
import { Octokit } from "@octokit/core";
import { paginateRest } from "@octokit/plugin-paginate-rest";
import { restEndpointMethods } from "@octokit/plugin-rest-endpoint-methods";

const MyOctokit = Octokit.plugin(paginateRest, restEndpointMethods);

const COMMENT_MARKER = "<!-- tensorzero-cla-bot -->";

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

    if (payload.repository?.owner?.login !== env.GITHUB_ORG) {
      return new Response("OK (skipped: wrong org)", { status: 200 });
    }

    const target = {
      owner: payload.repository.owner.login,
      repo: payload.repository.name,
    };

    const octokit = await app.getInstallationOctokit(
      Number(env.GITHUB_INSTALLATION_ID),
    );

    if (event === "pull_request") {
      if (!["opened", "reopened", "synchronize"].includes(payload.action)) {
        return new Response("OK (skipped action)", { status: 200 });
      }
      await evaluatePr(octokit, payload.pull_request, env, target);
      return new Response("OK", { status: 200 });
    }

    if (event === "merge_group") {
      if (payload.action !== "checks_requested") {
        return new Response("OK (skipped action)", { status: 200 });
      }
      await evaluateMergeGroup(octokit, payload.merge_group, env, target);
      return new Response("OK", { status: 200 });
    }

    // Fallback path for the merge queue: GitHub fires `check_suite` on the
    // synthetic queue branch (`gh-readonly-queue/...`) for every App with
    // Checks:write, even when the App's `merge_group` subscription is
    // misconfigured. We post the CLA check on the queue branch's head SHA
    // so branch protection unblocks.
    if (event === "check_suite") {
      if (
        payload.action !== "requested" &&
        payload.action !== "rerequested"
      ) {
        return new Response("OK (skipped action)", { status: 200 });
      }
      const branch = payload.check_suite.head_branch || "";
      if (!branch.startsWith("gh-readonly-queue/")) {
        return new Response("OK (skipped: not a queue branch)", {
          status: 200,
        });
      }
      const m = branch.match(/\/pr-(\d+)-/);
      if (!m) {
        return new Response("OK (skipped: cannot parse PR number)", {
          status: 200,
        });
      }
      const { data: pr } = await octokit.rest.pulls.get({
        owner: target.owner,
        repo: target.repo,
        pull_number: Number(m[1]),
      });
      await evaluateForQueueSha(
        octokit,
        pr,
        env,
        target,
        payload.check_suite.head_sha,
      );
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
        owner: target.owner,
        repo: target.repo,
        pull_number: payload.issue.number,
      });

      if (isSign) {
        await recordSignature(octokit, payload.comment, pr, env, target);
      }
      await evaluatePr(octokit, pr, env, target);
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

// `pulls.listCommits` is documented to return at most 250 commits per PR,
// regardless of pagination. Past that, we cannot reliably enumerate every
// contributor, so we fail closed instead of silently under-enforcing.
const MAX_COMMITS_INSPECTABLE = 250;

async function evaluatePr(octokit, pr, env, target) {
  const allowlist = parseAllowlist(env.ALLOWLIST);

  if (pr.commits > MAX_COMMITS_INSPECTABLE) {
    await postOversizedPrCheck(octokit, pr, env, target);
    return;
  }

  const commits = await octokit.paginate(octokit.rest.pulls.listCommits, {
    owner: target.owner,
    repo: target.repo,
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
    (u) => !isBotOrAllowlisted(u.login, allowlist),
  );

  // List comments once for both signature harvesting and the sticky comment.
  const comments = await octokit.paginate(octokit.rest.issues.listComments, {
    owner: target.owner,
    repo: target.repo,
    issue_number: pr.number,
    per_page: 100,
  });

  // Self-heal: any comment in this PR's thread whose body is exactly the
  // canonical sign phrase counts as a signature, even if it predates this
  // bot or was missed by a previous run. recordSignature is idempotent on
  // user id, so repeats are cheap no-ops.
  for (const c of comments) {
    if ((c.body || "").trim() !== env.SIGN_PHRASE) continue;
    if (!c.user?.id || !c.user?.login) continue;
    if (c.user.login.toLowerCase().endsWith("[bot]")) continue;
    await recordSignature(octokit, c, pr, env, target);
  }

  const signatures = await readSignatures(octokit, env, target);
  const signedIds = new Set(signatures.signedContributors.map((s) => s.id));
  const unsigned = required.filter((u) => !signedIds.has(u.id));

  const allSigned = unsigned.length === 0;
  await octokit.rest.checks.create({
    owner: target.owner,
    repo: target.repo,
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

  await upsertStickyComment(octokit, pr, unsigned, env, target, comments);
}

// Re-validate CLA on the merge-queue's synthetic head SHA. GitHub's branch
// protection runs required status checks on the queue branch (a fresh SHA),
// not the source PR's head, so without this the queue stalls. We do not edit
// the sticky comment or harvest signatures here — those happen on PR events.
async function evaluateMergeGroup(octokit, mergeGroup, env, target) {
  const headSha = mergeGroup.head_sha;
  const allowlist = parseAllowlist(env.ALLOWLIST);

  const { data: cmp } = await octokit.rest.repos.compareCommits({
    owner: target.owner,
    repo: target.repo,
    base: mergeGroup.base_sha,
    head: headSha,
  });

  if (cmp.total_commits > MAX_COMMITS_INSPECTABLE) {
    await octokit.rest.checks.create({
      owner: target.owner,
      repo: target.repo,
      name: env.CHECK_NAME,
      head_sha: headSha,
      status: "completed",
      conclusion: "action_required",
      details_url: env.CLA_DOC_URL,
      output: {
        title: "Merge group too large to verify CLA",
        summary: `This merge group includes ${cmp.total_commits} commits, exceeding the GitHub API's limit of ${MAX_COMMITS_INSPECTABLE} the CLA bot can enumerate.`,
      },
    });
    return;
  }

  const candidates = new Map();
  const addUser = (user) => {
    if (!user || !user.login || !user.id) return;
    if (user.login === "web-flow") return;
    candidates.set(user.id, { login: user.login, id: user.id });
  };
  for (const c of cmp.commits) {
    addUser(c.author);
    addUser(c.committer);
  }

  const required = [...candidates.values()].filter(
    (u) => !isBotOrAllowlisted(u.login, allowlist),
  );

  const signatures = await readSignatures(octokit, env, target);
  const signedIds = new Set(signatures.signedContributors.map((s) => s.id));
  const unsigned = required.filter((u) => !signedIds.has(u.id));

  const allSigned = unsigned.length === 0;
  await octokit.rest.checks.create({
    owner: target.owner,
    repo: target.repo,
    name: env.CHECK_NAME,
    head_sha: headSha,
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
}

// Same intent as evaluateMergeGroup, but used from the check_suite fallback
// path: derives required signers from the source PR's commit list (paginated)
// and posts the check on the queue branch's head SHA.
async function evaluateForQueueSha(octokit, pr, env, target, headSha) {
  const allowlist = parseAllowlist(env.ALLOWLIST);

  if (pr.commits > MAX_COMMITS_INSPECTABLE) {
    await octokit.rest.checks.create({
      owner: target.owner,
      repo: target.repo,
      name: env.CHECK_NAME,
      head_sha: headSha,
      status: "completed",
      conclusion: "action_required",
      details_url: env.CLA_DOC_URL,
      output: {
        title: "Pull request too large to verify CLA",
        summary: `This pull request has ${pr.commits} commits, exceeding the GitHub API's limit of ${MAX_COMMITS_INSPECTABLE} the CLA bot can enumerate.`,
      },
    });
    return;
  }

  const commits = await octokit.paginate(octokit.rest.pulls.listCommits, {
    owner: target.owner,
    repo: target.repo,
    pull_number: pr.number,
    per_page: 100,
  });

  const candidates = new Map();
  const addUser = (user) => {
    if (!user || !user.login || !user.id) return;
    if (user.login === "web-flow") return;
    candidates.set(user.id, { login: user.login, id: user.id });
  };
  addUser(pr.user);
  for (const c of commits) {
    addUser(c.author);
    addUser(c.committer);
  }

  const required = [...candidates.values()].filter(
    (u) => !isBotOrAllowlisted(u.login, allowlist),
  );

  const signatures = await readSignatures(octokit, env, target);
  const signedIds = new Set(signatures.signedContributors.map((s) => s.id));
  const unsigned = required.filter((u) => !signedIds.has(u.id));

  const allSigned = unsigned.length === 0;
  await octokit.rest.checks.create({
    owner: target.owner,
    repo: target.repo,
    name: env.CHECK_NAME,
    head_sha: headSha,
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
}

async function postOversizedPrCheck(octokit, pr, env, target) {
  const summary = [
    `This pull request has ${pr.commits} commits, which exceeds the GitHub API's per-PR limit of ${MAX_COMMITS_INSPECTABLE} commits the CLA bot can enumerate.`,
    "",
    "The bot cannot reliably verify that every commit author has signed the CLA, so this check fails closed.",
    "",
    "Please split this PR into smaller pieces, or contact the maintainers for manual review.",
  ].join("\n");

  await octokit.rest.checks.create({
    owner: target.owner,
    repo: target.repo,
    name: env.CHECK_NAME,
    head_sha: pr.head.sha,
    status: "completed",
    conclusion: "action_required",
    details_url: env.CLA_DOC_URL,
    output: {
      title: "Pull request too large to verify CLA",
      summary,
    },
  });

  const body = [
    COMMENT_MARKER,
    "",
    `⚠️ This pull request has **${pr.commits} commits**, exceeding GitHub's per-PR limit of ${MAX_COMMITS_INSPECTABLE} commits the CLA bot can enumerate. The bot cannot verify CLA coverage automatically.`,
    "",
    "Please split this PR into smaller pieces, or contact the maintainers for manual review.",
  ].join("\n");

  const comments = await octokit.paginate(octokit.rest.issues.listComments, {
    owner: target.owner,
    repo: target.repo,
    issue_number: pr.number,
    per_page: 100,
  });
  const appId = Number(env.GITHUB_APP_ID);
  const existing = comments.find(
    (c) =>
      c.performed_via_github_app?.id === appId &&
      (c.body || "").includes(COMMENT_MARKER),
  );

  if (existing) {
    if ((existing.body || "") === body) return;
    await octokit.rest.issues.updateComment({
      owner: target.owner,
      repo: target.repo,
      comment_id: existing.id,
      body,
    });
    return;
  }

  await octokit.rest.issues.createComment({
    owner: target.owner,
    repo: target.repo,
    issue_number: pr.number,
    body,
  });
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

async function upsertStickyComment(octokit, pr, unsigned, env, target, comments) {
  // Only treat a comment as the sticky comment if this GitHub App authored it.
  // Otherwise a contributor could post the marker themselves and trick the bot
  // into trying to edit a comment it doesn't own (which GitHub rejects).
  const appId = Number(env.GITHUB_APP_ID);
  const existing = comments.find(
    (c) =>
      c.performed_via_github_app?.id === appId &&
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
      owner: target.owner,
      repo: target.repo,
      comment_id: existing.id,
      body,
    });
    return;
  }

  await octokit.rest.issues.createComment({
    owner: target.owner,
    repo: target.repo,
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

async function recordSignature(octokit, comment, pr, env, target) {
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
    const { ref, signatures } = await readSignaturesWithSha(
      octokit,
      env,
      target,
    );

    if (signatures.signedContributors.some((s) => s.id === newEntry.id)) {
      return;
    }
    signatures.signedContributors.push(newEntry);

    try {
      await commitSignatures(octokit, env, target, ref, signatures, newEntry);
      return;
    } catch (err) {
      // 422: createCommit rejected because base tree/parent is stale.
      // 409: updateRef rejected as non-fast-forward (concurrent signer or
      // duplicate webhook delivery). Both indicate a race we should retry.
      if ((err.status === 422 || err.status === 409) && attempt < maxAttempts) {
        await sleep(100 + Math.floor(Math.random() * 200));
        continue;
      }
      throw err;
    }
  }
}

async function readSignatures(octokit, env, target) {
  const { signatures } = await readSignaturesWithSha(octokit, env, target);
  return signatures;
}

async function readSignaturesWithSha(octokit, env, target) {
  let ref;
  try {
    const { data } = await octokit.rest.git.getRef({
      owner: target.owner,
      repo: target.repo,
      ref: `heads/${env.CLA_BRANCH}`,
    });
    ref = data;
  } catch (err) {
    if (err.status !== 404) throw err;
    ref = await bootstrapClaBranch(octokit, env, target);
  }

  let signatures;
  try {
    const { data: file } = await octokit.rest.repos.getContent({
      owner: target.owner,
      repo: target.repo,
      path: env.CLA_SIGNATURES_PATH,
      ref: env.CLA_BRANCH,
    });
    signatures = JSON.parse(decodeBase64Utf8(file.content));
  } catch (err) {
    if (err.status !== 404) throw err;
    signatures = { signedContributors: [] };
  }

  return { ref, signatures };
}

async function bootstrapClaBranch(octokit, env, target) {
  const initial = JSON.stringify({ signedContributors: [] }, null, 2) + "\n";
  const { data: blob } = await octokit.rest.git.createBlob({
    owner: target.owner,
    repo: target.repo,
    content: initial,
    encoding: "utf-8",
  });
  const { data: tree } = await octokit.rest.git.createTree({
    owner: target.owner,
    repo: target.repo,
    tree: [
      {
        path: env.CLA_SIGNATURES_PATH,
        mode: "100644",
        type: "blob",
        sha: blob.sha,
      },
    ],
  });
  // Orphan commit (no parents) — keeps the cla-signatures branch isolated
  // from the repo's main history.
  const { data: commit } = await octokit.rest.git.createCommit({
    owner: target.owner,
    repo: target.repo,
    message: "Initialize CLA signatures",
    tree: tree.sha,
    parents: [],
  });
  try {
    const { data: ref } = await octokit.rest.git.createRef({
      owner: target.owner,
      repo: target.repo,
      ref: `refs/heads/${env.CLA_BRANCH}`,
      sha: commit.sha,
    });
    return ref;
  } catch (err) {
    // 422 here means another concurrent webhook (or a redelivery) created
    // the branch between our getRef 404 and this createRef. Fetch and use
    // whatever they wrote; our orphan blob/tree/commit is unreachable and
    // GitHub will GC it.
    if (err.status !== 422) throw err;
    const { data: ref } = await octokit.rest.git.getRef({
      owner: target.owner,
      repo: target.repo,
      ref: `heads/${env.CLA_BRANCH}`,
    });
    return ref;
  }
}

async function commitSignatures(octokit, env, target, ref, signatures, newEntry) {
  const { data: parentCommit } = await octokit.rest.git.getCommit({
    owner: target.owner,
    repo: target.repo,
    commit_sha: ref.object.sha,
  });

  const newContent = JSON.stringify(signatures, null, 2) + "\n";
  const { data: blob } = await octokit.rest.git.createBlob({
    owner: target.owner,
    repo: target.repo,
    content: newContent,
    encoding: "utf-8",
  });

  const { data: tree } = await octokit.rest.git.createTree({
    owner: target.owner,
    repo: target.repo,
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
    owner: target.owner,
    repo: target.repo,
    message: `Sign CLA: @${newEntry.name} (#${newEntry.pullRequestNo})`,
    tree: tree.sha,
    parents: [parentCommit.sha],
  });

  await octokit.rest.git.updateRef({
    owner: target.owner,
    repo: target.repo,
    ref: `heads/${env.CLA_BRANCH}`,
    sha: commit.sha,
    force: false,
  });
}

// --- Helpers ---

function isBotOrAllowlisted(login, allowlist) {
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
