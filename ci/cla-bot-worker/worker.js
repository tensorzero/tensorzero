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

    // Org filter (case-insensitive — guard against a typo'd GITHUB_ORG env var
    // or any future variation in payload casing).
    if (
      payload.repository?.owner?.login?.toLowerCase() !==
      env.GITHUB_ORG?.toLowerCase()
    ) {
      return new Response("OK (skipped: wrong org)", { status: 200 });
    }

    const target = {
      owner: payload.repository.owner.login,
      repo: payload.repository.name,
    };

    // Installation Octokit is minted lazily per branch — webhook events we
    // skip don't need a token, so we avoid an authenticated request to GitHub
    // on every drive-by delivery (`check_suite.requested` on every push,
    // unhandled events, wrong actions).
    const installationOctokit = () =>
      app.getInstallationOctokit(Number(env.GITHUB_INSTALLATION_ID));

    if (event === "pull_request") {
      if (!["opened", "reopened", "synchronize"].includes(payload.action)) {
        return new Response("OK (skipped action)", { status: 200 });
      }
      const octokit = await installationOctokit();
      await evaluatePr(octokit, payload.pull_request, env, target);
      return new Response("OK", { status: 200 });
    }

    if (event === "merge_group") {
      if (payload.action !== "checks_requested") {
        return new Response("OK (skipped action)", { status: 200 });
      }
      const octokit = await installationOctokit();
      await evaluateQueueRange(
        octokit,
        env,
        target,
        payload.merge_group.base_sha,
        payload.merge_group.head_sha,
      );
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
      // The queue branch name format is
      //   gh-readonly-queue/<target_branch>/pr-<N>-<base_sha>
      // The trailing 40-char hex is the base branch's SHA at queue creation,
      // which is what we want to compare against. We do NOT key off the PR
      // number (queue groups can batch multiple PRs; that branch name only
      // mentions one of them).
      const m = branch.match(/-([0-9a-f]{40})$/);
      if (!m) {
        return new Response("OK (skipped: cannot parse base SHA)", {
          status: 200,
        });
      }
      const octokit = await installationOctokit();
      await evaluateQueueRange(
        octokit,
        env,
        target,
        m[1],
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

      const octokit = await installationOctokit();
      const { data: pr } = await octokit.rest.pulls.get({
        owner: target.owner,
        repo: target.repo,
        pull_number: payload.issue.number,
      });

      // Explicit recordSignature here is a safety net for read-after-write
      // lag on the listComments API used inside evaluatePr's harvest loop —
      // GitHub may not yet return the just-posted comment when we list. The
      // harvest loop will skip duplicates via local signedIds tracking.
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

  // List comments once. Used for both signature harvesting (always) and the
  // sticky-comment upsert (in either the unresolved or normal path).
  const comments = await octokit.paginate(octokit.rest.issues.listComments, {
    owner: target.owner,
    repo: target.repo,
    issue_number: pr.number,
    per_page: 100,
  });

  // Harvest signatures BEFORE the unresolved-commits gate: a contributor who
  // posted the canonical phrase has signed the CLA regardless of whether
  // every commit's email is currently linked. Once they fix their commits,
  // their sig is already on file and they don't need to re-comment.
  const signatures = await readSignatures(octokit, env, target);
  const signedIds = new Set(signatures.signedContributors.map((s) => s.id));
  for (const c of comments) {
    if ((c.body || "").trim() !== env.SIGN_PHRASE) continue;
    if (!c.user?.id || !c.user?.login) continue;
    if (c.user.login.toLowerCase().endsWith("[bot]")) continue;
    if (signedIds.has(c.user.id)) continue;
    await recordSignature(octokit, c, pr, env, target);
    signedIds.add(c.user.id);
  }

  const unresolved = findUnresolvedCommits(commits);
  if (unresolved.length > 0) {
    await postUnresolvedCommitsCheck(
      octokit,
      target,
      env,
      pr.head.sha,
      unresolved,
    );
    await upsertBotComment(
      octokit,
      target,
      pr.number,
      comments,
      env,
      [COMMENT_MARKER, "", buildUnresolvedSummary(unresolved)].join("\n"),
    );
    return;
  }

  const candidates = collectCandidates(commits, pr.user);
  const required = [...candidates.values()].filter(
    (u) => !isBotOrAllowlisted(u.login, allowlist),
  );
  const unsigned = required.filter((u) => !signedIds.has(u.id));

  await upsertCheckRun(octokit, target, env, pr.head.sha, {
    conclusion: unsigned.length === 0 ? "success" : "action_required",
    output: {
      title:
        unsigned.length === 0
          ? "All contributors have signed the CLA"
          : "CLA signature required",
      summary: buildCheckSummary(unsigned, env),
    },
  });

  await upsertStickyComment(octokit, pr, unsigned, env, target, comments);
}

// Re-validate CLA on a merge-queue branch's range (base_sha..head_sha). The
// queue may batch multiple PRs into one branch, so we always derive
// contributors from the actual commits in the range — not from any one PR.
// GitHub branch protection runs required checks on the queue branch's SHA,
// not the source PR's head, so without this the queue stalls. We do not
// touch the sticky comment or harvest signatures here — those are PR-event
// concerns.
async function evaluateQueueRange(octokit, env, target, baseSha, headSha) {
  const allowlist = parseAllowlist(env.ALLOWLIST);

  const { data: cmp } = await octokit.rest.repos.compareCommits({
    owner: target.owner,
    repo: target.repo,
    base: baseSha,
    head: headSha,
  });

  if (cmp.total_commits > MAX_COMMITS_INSPECTABLE) {
    await upsertCheckRun(octokit, target, env, headSha, {
      conclusion: "action_required",
      output: {
        title: "Merge group too large to verify CLA",
        summary: `This merge group includes ${cmp.total_commits} commits, exceeding the GitHub API's limit of ${MAX_COMMITS_INSPECTABLE} the CLA bot can enumerate.`,
      },
    });
    return;
  }

  const unresolved = findUnresolvedCommits(cmp.commits);
  if (unresolved.length > 0) {
    await postUnresolvedCommitsCheck(octokit, target, env, headSha, unresolved);
    return;
  }

  const candidates = collectCandidates(cmp.commits);
  const required = [...candidates.values()].filter(
    (u) => !isBotOrAllowlisted(u.login, allowlist),
  );

  const signatures = await readSignatures(octokit, env, target);
  const signedIds = new Set(signatures.signedContributors.map((s) => s.id));
  const unsigned = required.filter((u) => !signedIds.has(u.id));

  await upsertCheckRun(octokit, target, env, headSha, {
    conclusion: unsigned.length === 0 ? "success" : "action_required",
    output: {
      title:
        unsigned.length === 0
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

  await upsertCheckRun(octokit, target, env, pr.head.sha, {
    conclusion: "action_required",
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
  await upsertBotComment(octokit, target, pr.number, comments, env, body);
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
  // If everyone was already signed when the PR was opened, stay silent —
  // the green Check Run is enough. Only edit the comment if we previously
  // posted one (i.e. someone went from unsigned to signed in this PR).
  const appId = Number(env.GITHUB_APP_ID);
  const hasExistingBotComment = comments.some(
    (c) =>
      c.performed_via_github_app?.id === appId &&
      (c.body || "").includes(COMMENT_MARKER),
  );
  if (unsigned.length === 0 && !hasExistingBotComment) return;

  await upsertBotComment(
    octokit,
    target,
    pr.number,
    comments,
    env,
    renderStickyBody(unsigned, env),
  );
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

  // Defensive: a manually-edited or partially-truncated file might be missing
  // the array entirely. Treat as empty rather than crashing on .map/.some.
  if (!Array.isArray(signatures.signedContributors)) {
    signatures.signedContributors = [];
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
    // 422/409 here means another concurrent webhook (or a redelivery) created
    // the branch between our getRef 404 and this createRef. Fetch and use
    // whatever they wrote; our orphan blob/tree/commit is unreachable and
    // GitHub will GC it.
    if (err.status !== 422 && err.status !== 409) throw err;
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

function collectCandidates(commits, prUser) {
  const candidates = new Map();
  const addUser = (user) => {
    if (!user || !user.login || !user.id) return;
    // GitHub's web-flow committer represents browser-side commits (e.g.
    // squash merges). It's a synthetic identity, not a real contributor.
    if (user.login === "web-flow") return;
    candidates.set(user.id, { login: user.login, id: user.id });
  };
  if (prUser) addUser(prUser);
  for (const c of commits) {
    addUser(c.author);
    addUser(c.committer);
  }
  return candidates;
}

// A commit whose top-level `author` or `committer` field is null is one whose
// raw git author/committer email is not linked to any GitHub account. We
// cannot tell who signed the CLA for that commit, so we fail closed instead
// of silently treating the commit as not requiring a signature.
//
// When both author and committer are unresolved with the same email (the
// common local-`git commit` case), we list the commit once with role
// "author/committer" rather than emitting two redundant lines.
function findUnresolvedCommits(commits) {
  const unresolved = [];
  for (const c of commits) {
    const aMissing = c.author === null;
    const cMissing = c.committer === null;
    if (!aMissing && !cMissing) continue;

    const aName = c.commit?.author?.name || "(unknown)";
    const aEmail = c.commit?.author?.email || "(unknown)";
    const cName = c.commit?.committer?.name || "(unknown)";
    const cEmail = c.commit?.committer?.email || "(unknown)";

    if (aMissing && cMissing && aEmail === cEmail && aName === cName) {
      unresolved.push({
        sha: c.sha,
        role: "author/committer",
        name: aName,
        email: aEmail,
      });
      continue;
    }
    if (aMissing) {
      unresolved.push({ sha: c.sha, role: "author", name: aName, email: aEmail });
    }
    if (cMissing) {
      unresolved.push({
        sha: c.sha,
        role: "committer",
        name: cName,
        email: cEmail,
      });
    }
  }
  return unresolved;
}

// Strip characters from user-controlled (`git config`) name/email that would
// break inline-code rendering or escape into surrounding markdown: backticks
// (close the code span) and CR/LF (break the list item across lines).
function safeForCodeSpan(s) {
  return (s || "(unknown)").replace(/[`\r\n]/g, "");
}

function buildUnresolvedSummary(unresolved) {
  const lines = unresolved
    .slice(0, 25)
    .map(
      (u) =>
        `- \`${u.sha.slice(0, 7)}\` ${u.role}: \`${safeForCodeSpan(u.name)} <${safeForCodeSpan(u.email)}>\``,
    );
  if (unresolved.length > 25) {
    lines.push(`- … and ${unresolved.length - 25} more`);
  }
  return [
    "The CLA bot cannot verify CLA coverage for this pull request because some commits are attributed to email addresses not linked to a GitHub account. Without a GitHub identity for those commits, the bot cannot determine whether the contributor has signed the CLA.",
    "",
    "**Affected commits:**",
    "",
    ...lines,
    "",
    "**To resolve:**",
    "",
    "- Each affected contributor adds the email above to their GitHub account: https://github.com/settings/emails (recommended), **or**",
    "- Re-author the affected commits with a GitHub-linked email and force-push.",
    "",
    "Once the commits resolve to GitHub users, push or comment `recheck` to re-evaluate.",
  ].join("\n");
}

// Update the existing CLA Check Run on this SHA if we already posted one,
// otherwise create a new one. Without this, every webhook redelivery and
// every recheck would stack up duplicate `cla` runs in the PR's checks tab.
async function upsertCheckRun(octokit, target, env, headSha, params) {
  const appId = Number(env.GITHUB_APP_ID);
  const { data } = await octokit.rest.checks.listForRef({
    owner: target.owner,
    repo: target.repo,
    ref: headSha,
    check_name: env.CHECK_NAME,
    app_id: appId,
    filter: "latest",
    per_page: 1,
  });

  const base = {
    status: "completed",
    details_url: env.CLA_DOC_URL,
    ...params,
  };

  if (data.check_runs.length > 0) {
    await octokit.rest.checks.update({
      owner: target.owner,
      repo: target.repo,
      check_run_id: data.check_runs[0].id,
      ...base,
    });
    return;
  }

  await octokit.rest.checks.create({
    owner: target.owner,
    repo: target.repo,
    name: env.CHECK_NAME,
    head_sha: headSha,
    ...base,
  });
}

async function postUnresolvedCommitsCheck(
  octokit,
  target,
  env,
  headSha,
  unresolved,
) {
  await upsertCheckRun(octokit, target, env, headSha, {
    conclusion: "action_required",
    output: {
      title: "Cannot verify CLA: commits with unlinked email addresses",
      summary: buildUnresolvedSummary(unresolved),
    },
  });
}

// Single upsert path for the bot's sticky comment. Finds the comment authored
// by this GitHub App that contains the marker, updates it if the body
// changed, creates one otherwise. Filtering by `performed_via_github_app.id`
// guards against a contributor pasting the marker into their own comment to
// trick the bot into trying to edit a comment it doesn't own.
async function upsertBotComment(octokit, target, prNumber, comments, env, body) {
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
    issue_number: prNumber,
    body,
  });
}

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
