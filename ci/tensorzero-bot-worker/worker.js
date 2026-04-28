// Cloudflare Worker that receives GitHub webhook events and runs PR-housekeeping
// automation across every repo in the tensorzero org. Replaces the
// `force-merge-queue.yml` and `label-merge-conflicts.yml` workflows.
//
// Required secrets (set via `wrangler secret put`):
//   GITHUB_APP_ID          - GitHub App ID
//   GITHUB_APP_PRIVATE_KEY - GitHub App private key (PKCS#8 PEM)
//   GITHUB_INSTALLATION_ID - GitHub App installation ID for the tensorzero org
//   GITHUB_WEBHOOK_SECRET  - webhook secret configured in the GitHub App
//
// Required vars (set in wrangler.toml):
//   GITHUB_ORG
//   FORCE_MERGE_QUEUE_LABEL, FORCE_MERGE_QUEUE_STATUS_CONTEXT
//   GENERAL_CHECK_RUN_NAME
//   DIRTY_LABEL, DIRTY_LABEL_COLOR, DIRTY_LABEL_DESCRIPTION

import { App } from "@octokit/app";
import { Octokit } from "@octokit/core";
import { paginateRest } from "@octokit/plugin-paginate-rest";
import { restEndpointMethods } from "@octokit/plugin-rest-endpoint-methods";

const MyOctokit = Octokit.plugin(paginateRest, restEndpointMethods);

export default {
  async fetch(request, env, ctx) {
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
    // on every drive-by delivery.
    const installationOctokit = () =>
      app.getInstallationOctokit(Number(env.GITHUB_INSTALLATION_ID));

    if (event === "pull_request") {
      if (payload.action === "labeled") {
        if (payload.label?.name !== env.FORCE_MERGE_QUEUE_LABEL) {
          return new Response("OK (skipped label)", { status: 200 });
        }
        const octokit = await installationOctokit();
        await forceMergeQueue(octokit, payload.pull_request, target, env, ctx);
        return new Response("OK", { status: 200 });
      }
      if (
        payload.action === "opened" ||
        payload.action === "reopened" ||
        payload.action === "synchronize"
      ) {
        // Defer to ctx.waitUntil: getMergeableState polls up to ~10s, which
        // is GitHub's webhook delivery deadline. Acknowledging immediately
        // and continuing in the background keeps deliveries marked
        // successful even when GitHub is slow to compute mergeable state.
        //
        // Wrapped in withRetry: GitHub doesn't redeliver acknowledged
        // webhooks, so transient 5xx/429 on pulls.get / addLabels /
        // removeLabel would otherwise leave the label stale until an
        // unrelated later event. Retry with backoff before giving up.
        const prNumber = payload.pull_request.number;
        ctx.waitUntil(
          withRetry(
            async () => {
              const octokit = await installationOctokit();
              await labelMergeConflicts(octokit, prNumber, target, env);
            },
            `labelMergeConflicts(PR #${prNumber})`,
          ).catch((err) => {
            console.error(
              `labelMergeConflicts: exhausted retries for PR #${prNumber}:`,
              err?.status,
              err?.message,
            );
          }),
        );
        return new Response("OK", { status: 200 });
      }
      // unlabeled, closed, etc.: no-op. We deliberately do nothing on
      // unlabeled — once the success commit status is posted it stands until
      // a new commit pushes a fresh general.yml status that supersedes it.
      return new Response("OK (skipped action)", { status: 200 });
    }

    if (event === "push") {
      // Tag pushes (refs/tags/*) cannot be the base of any PR; skip without
      // even minting an installation token. Branch deletions push with
      // all-zeros `after`; same treatment.
      if (
        !payload.ref?.startsWith("refs/heads/") ||
        /^0+$/.test(payload.after || "")
      ) {
        return new Response("OK (skipped: not a branch push)", { status: 200 });
      }
      // Defer: a push to a base branch fans out across every open PR
      // targeting it. The round-based fan-out (see
      // labelMergeConflictsForPushedRef) takes ~7-12s for tensorzero scale
      // — comfortably past GitHub's 10s webhook deadline. ctx.waitUntil
      // keeps the worker alive in the background after we ack.
      ctx.waitUntil(
        (async () => {
          // Retry just the token mint — `pulls.list` and the per-PR fan-out
          // already have their own retries inside
          // labelMergeConflictsForPushedRef, so wrapping the whole closure
          // would double-count.
          const octokit = await withRetry(
            installationOctokit,
            "installationOctokit (push)",
          );
          await labelMergeConflictsForPushedRef(octokit, payload, target, env);
        })().catch((err) => {
          console.error(
            "labelMergeConflictsForPushedRef: failed:",
            err?.status,
            err?.message,
          );
        }),
      );
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

// --- Force merge queue ---

// Triggered when a PR receives the `force-add-to-merge-queue` label. Posts a
// success commit status for `check-all-general-jobs-passed` so the PR can
// enter the merge queue without waiting on general.yml. The merge queue still
// runs the real check on the queue branch's SHA, so this is always safe.
async function forceMergeQueue(octokit, pr, target, env, ctx) {
  // If the PR was closed/merged between the labeled event and our
  // processing, posting a status on the (now stale) head SHA is harmless
  // but pointless. Skip.
  if (pr.state !== "open") return;

  // Post the success status FIRST. This is the load-bearing step that
  // unblocks merge-queue entry. Wrapped in withRetry so a transient
  // 5xx/429 recovers in seconds instead of waiting for GitHub's webhook
  // redelivery cycle (~1+ minute). On persistent failure, we still throw
  // → Cloudflare 500 → GitHub redelivers as the eventual fallback.
  //
  // Note: if the contributor pushes during this flow, the new head SHA
  // gets its own general.yml run and our success status lands on the old
  // (no longer head) SHA. That's the correct outcome — pushing should
  // reset the gate.
  await withRetry(
    () =>
      octokit.rest.repos.createCommitStatus({
        owner: target.owner,
        repo: target.repo,
        sha: pr.head.sha,
        context: env.FORCE_MERGE_QUEUE_STATUS_CONTEXT,
        state: "success",
        description: "Forced via force-add-to-merge-queue label",
        target_url: `https://github.com/${target.owner}/${target.repo}/pull/${pr.number}`,
      }),
    `createCommitStatus(PR #${pr.number})`,
  );

  // Best-effort: if there's a failed `general.yml` run for this SHA, restart
  // it so the real check eventually flips to success without the contributor
  // having to push an empty commit. GitHub's UI shows the failed run last
  // even though our success status is posted, which is confusing —
  // restarting resolves that. Done in the background so we never block the
  // status post on Actions API latency.
  ctx.waitUntil(rerunFailedGeneralRuns(octokit, pr.head.sha, target, env));
}

async function rerunFailedGeneralRuns(octokit, headSha, target, env) {
  try {
    const { data: checks } = await octokit.rest.checks.listForRef({
      owner: target.owner,
      repo: target.repo,
      ref: headSha,
      check_name: env.GENERAL_CHECK_RUN_NAME,
      per_page: 100,
    });
    const seenRunIds = new Set();
    const reruns = [];
    for (const c of checks.check_runs) {
      if (c.conclusion !== "failure") continue;
      const m = (c.details_url || "").match(/\/actions\/runs\/(\d+)/);
      if (!m) continue;
      const runId = Number(m[1]);
      if (seenRunIds.has(runId)) continue;
      seenRunIds.add(runId);
      reruns.push(
        octokit.rest.actions
          .reRunWorkflow({
            owner: target.owner,
            repo: target.repo,
            run_id: runId,
          })
          .catch((e) => {
            console.error(
              `forceMergeQueue: reRunWorkflow ${runId} failed:`,
              e?.status,
              e?.message,
            );
          }),
      );
    }
    await Promise.allSettled(reruns);
  } catch (err) {
    if (err?.status !== 404) {
      console.error(
        "forceMergeQueue: listForRef failed:",
        err?.status,
        err?.message,
      );
    }
  }
}

// --- Label merge conflicts ---

async function labelMergeConflicts(octokit, prNumber, target, env) {
  const pr = await getMergeableState(octokit, target, prNumber);
  if (!pr) return; // gave up; next webhook event will re-evaluate
  // pr.state can be "closed" if the PR was closed between the event firing
  // and our processing — skip those.
  if (pr.state !== "open") return;
  if (pr.mergeable === false) {
    await ensureLabelPresent(octokit, target, env, prNumber);
  } else if (pr.mergeable === true) {
    await ensureLabelAbsent(octokit, target, env, prNumber);
  }
  // mergeable === null after polling: skip; next event re-evaluates.
}

async function labelMergeConflictsForPushedRef(octokit, payload, target, env) {
  // Pushes to a PR's head branch are already covered by
  // pull_request.synchronize. Pushes to a base branch can flip mergeable for
  // every PR targeting it — list those and re-evaluate.
  const ref = payload.ref || "";
  // Tag pushes (refs/tags/*) cannot be the base of any PR. Skip.
  if (!ref.startsWith("refs/heads/")) return;
  const baseRef = ref.slice("refs/heads/".length);
  if (!baseRef) return;

  // Branch deletion fires push events with after === all-zeros; skip.
  if (/^0+$/.test(payload.after || "")) return;

  // Retry pulls.list specifically — if it transiently fails on a busy
  // base-branch push, the entire fan-out drops with no GitHub redelivery
  // (waitUntil already 200'd) and every PR's label stays stale until the
  // next event.
  const prs = await withRetry(
    () =>
      octokit.paginate(octokit.rest.pulls.list, {
        owner: target.owner,
        repo: target.repo,
        state: "open",
        base: baseRef,
        per_page: 100,
      }),
    `pulls.list(base=${baseRef})`,
  );

  // Round-based fan-out: each round makes a single pulls.get per pending
  // PR (with concurrency cap), labels the ones with definitive mergeable,
  // and carries over the null ones to the next round. This amortizes the
  // mergeable-poll wait across ALL pending PRs instead of paying a 3s
  // backoff per batch sequentially. For 100 PRs:
  //   round 1 (immediate): 5 batches of 20 × ~500ms ≈ 2.5s
  //   round 2 (after 1s):  ~50% still null → ~1.5s
  //   round 3 (after 2s):  ~10% still null → ~0.5s
  //   total ≈ 7-9s including wait, well under the ~30s ctx.waitUntil budget.
  // PRs whose mergeable is still null after 3 rounds are dropped this
  // round; the next webhook event will re-evaluate them.
  let pending = prs.map((p) => p.number);
  const roundWaits = [0, 1000, 2000];
  for (let i = 0; i < roundWaits.length && pending.length > 0; i++) {
    if (roundWaits[i]) await sleep(roundWaits[i]);
    pending = await runMergeableRound(octokit, target, env, pending);
  }
}

// One pulls.get per PR + at most one label upsert. Returns the PR numbers
// whose mergeable was still null (need another round).
async function runMergeableRound(octokit, target, env, prNumbers, concurrency = 20) {
  const stillPending = [];
  const evaluate = async (n) => {
    const { data } = await octokit.rest.pulls.get({
      owner: target.owner,
      repo: target.repo,
      pull_number: n,
    });
    if (data.state !== "open") return null;
    if (data.mergeable === true) {
      await ensureLabelAbsent(octokit, target, env, n);
      return null;
    }
    if (data.mergeable === false) {
      await ensureLabelPresent(octokit, target, env, n);
      return null;
    }
    return n; // mergeable === null → carry over
  };
  for (let i = 0; i < prNumbers.length; i += concurrency) {
    const batch = prNumbers.slice(i, i + concurrency);
    const results = await Promise.allSettled(batch.map(evaluate));
    for (let j = 0; j < results.length; j++) {
      const r = results[j];
      if (r.status === "rejected") {
        // Transient errors (network, 5xx, 429) on pulls.get or label
        // upsert: re-queue to the next round so we get up to 3 attempts
        // (0+1+2s backoff). After the last round, persistent errors are
        // dropped — the next webhook event will re-evaluate.
        console.error(
          "runMergeableRound: PR",
          batch[j],
          "failed (will retry next round):",
          r.reason,
        );
        stillPending.push(batch[j]);
      } else if (r.value !== null) {
        stillPending.push(r.value);
      }
    }
  }
  return stillPending;
}

// GitHub computes `mergeable` lazily and returns null on the first read
// after a relevant change. Poll up to 5 times (~10s total) before giving
// up. Used by single-PR paths (synchronize/opened/reopened); the push
// fan-out has its own round-based polling that amortizes the wait.
async function getMergeableState(octokit, target, prNumber) {
  const delays = [0, 1000, 2000, 3000, 4000];
  for (const d of delays) {
    if (d) await sleep(d);
    const { data } = await octokit.rest.pulls.get({
      owner: target.owner,
      repo: target.repo,
      pull_number: prNumber,
    });
    if (data.mergeable !== null) return data;
  }
  return null;
}

async function ensureLabelPresent(octokit, target, env, prNumber) {
  try {
    await octokit.rest.issues.addLabels({
      owner: target.owner,
      repo: target.repo,
      issue_number: prNumber,
      labels: [env.DIRTY_LABEL],
    });
  } catch (err) {
    // GitHub returns 422 ("Validation Failed") when the label name doesn't
    // exist on the repo, and 404 in some edge cases. Either way: create the
    // label, then retry. Any other status is a real failure — propagate.
    if (err.status !== 404 && err.status !== 422) throw err;
    await octokit.rest.issues
      .createLabel({
        owner: target.owner,
        repo: target.repo,
        name: env.DIRTY_LABEL,
        color: env.DIRTY_LABEL_COLOR,
        description: env.DIRTY_LABEL_DESCRIPTION,
      })
      .catch((e) => {
        // 422 here = already exists (lost a race with a concurrent webhook).
        if (e.status !== 422) throw e;
      });
    await octokit.rest.issues.addLabels({
      owner: target.owner,
      repo: target.repo,
      issue_number: prNumber,
      labels: [env.DIRTY_LABEL],
    });
  }
}

async function ensureLabelAbsent(octokit, target, env, prNumber) {
  await octokit.rest.issues
    .removeLabel({
      owner: target.owner,
      repo: target.repo,
      issue_number: prNumber,
      name: env.DIRTY_LABEL,
    })
    .catch((e) => {
      // 404 = label wasn't on this PR (or label doesn't exist on the repo
      // yet). Either way, the desired state is satisfied.
      if (e.status !== 404) throw e;
    });
}

// --- Helpers ---

// Retry a one-shot async operation on transient failures: network errors
// (no status), 5xx, 429, and rate-limit-style 403s. Non-transient errors
// (4xx other than 429, permission-style 403s, etc.) throw immediately. Used
// to harden waitUntil-deferred work that GitHub won't redeliver after we
// ack with 200.
async function withRetry(fn, label, maxAttempts = 3) {
  let lastErr;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (attempt > 0) await sleep(1000 * attempt); // 1s, 2s
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (!isTransientError(err)) throw err;
      console.error(
        `withRetry(${label}): attempt ${attempt + 1} failed (status=${err?.status}):`,
        err?.message,
      );
    }
  }
  throw lastErr;
}

// GitHub returns 403 for both genuine permission errors and secondary rate
// limits. Heuristic: a rate-limit-style 403 carries a `Retry-After` header
// or a body message containing "rate limit" or "abuse". Permission-style
// 403s have neither, so retrying them just wastes attempts.
function isTransientError(err) {
  const status = err?.status;
  if (!status) return true; // network error / no response
  if (status >= 500) return true;
  if (status === 429) return true;
  if (status === 403) {
    const headers = err?.response?.headers || {};
    if (headers["retry-after"]) return true;
    const message = (
      err?.response?.data?.message ||
      err?.message ||
      ""
    ).toLowerCase();
    if (message.includes("rate limit") || message.includes("abuse")) {
      return true;
    }
  }
  return false;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
