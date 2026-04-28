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
        await forceMergeQueue(octokit, payload.pull_request, target, env);
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
        ctx.waitUntil(
          (async () => {
            const octokit = await installationOctokit();
            await labelMergeConflicts(
              octokit,
              payload.pull_request.number,
              target,
              env,
            );
          })(),
        );
        return new Response("OK", { status: 200 });
      }
      // unlabeled, closed, etc.: no-op. We deliberately do nothing on
      // unlabeled — once the success commit status is posted it stands until
      // a new commit pushes a fresh general.yml status that supersedes it.
      return new Response("OK (skipped action)", { status: 200 });
    }

    if (event === "push") {
      // Defer: a push to a base branch fans out to every open PR targeting
      // it, each with up to ~10s of mergeable polling. Far past GitHub's
      // 10s webhook deadline. ctx.waitUntil keeps the worker alive in the
      // background after we ack.
      ctx.waitUntil(
        (async () => {
          const octokit = await installationOctokit();
          await labelMergeConflictsForPushedRef(octokit, payload, target, env);
        })(),
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
async function forceMergeQueue(octokit, pr, target, env) {
  // Best-effort: if there's a failed `general.yml` run for this SHA, restart
  // it so the real check eventually flips to success without the contributor
  // having to push an empty commit. GitHub's UI shows the failed run last
  // even though our success status is posted, which is confusing — restarting
  // resolves that. Errors here are not fatal; the success status is the
  // important part.
  try {
    const { data: checks } = await octokit.rest.checks.listForRef({
      owner: target.owner,
      repo: target.repo,
      ref: pr.head.sha,
      check_name: env.GENERAL_CHECK_RUN_NAME,
      per_page: 100,
    });
    const failed = checks.check_runs.filter(
      (c) => c.conclusion === "failure",
    );
    const restartedRunIds = new Set();
    for (const c of failed) {
      const m = (c.details_url || "").match(/\/actions\/runs\/(\d+)/);
      if (!m) continue;
      const runId = Number(m[1]);
      if (restartedRunIds.has(runId)) continue;
      restartedRunIds.add(runId);
      await octokit.rest.actions
        .reRunWorkflow({
          owner: target.owner,
          repo: target.repo,
          run_id: runId,
        })
        .catch(() => {
          // Best-effort. GitHub rejects re-runs on certain workflow states
          // (e.g. already running, too old). Ignore and proceed.
        });
    }
  } catch (err) {
    // Same: best-effort. Proceed to the status post.
    if (err.status !== 404) {
      // Don't swallow non-404 entirely — log via the thrown response if it
      // matters. Practically: continue.
    }
  }

  await octokit.rest.repos.createCommitStatus({
    owner: target.owner,
    repo: target.repo,
    sha: pr.head.sha,
    context: env.FORCE_MERGE_QUEUE_STATUS_CONTEXT,
    state: "success",
    description: "Forced via force-add-to-merge-queue label",
    target_url: `https://github.com/${target.owner}/${target.repo}/pull/${pr.number}`,
  });
}

// --- Label merge conflicts ---

async function labelMergeConflicts(octokit, prNumber, target, env, pollAttempts) {
  const pr = await getMergeableState(octokit, target, prNumber, pollAttempts);
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
  const baseRef = payload.ref?.replace(/^refs\/heads\//, "");
  if (!baseRef) return;

  // Tag/branch deletion fires push events with after === all-zeros; skip.
  if (/^0+$/.test(payload.after || "")) return;

  const prs = await octokit.paginate(octokit.rest.pulls.list, {
    owner: target.owner,
    repo: target.repo,
    state: "open",
    base: baseRef,
    per_page: 100,
  });
  // Cloudflare cancels ctx.waitUntil after ~30s. With the default 10s poll
  // budget, a base-branch push to a repo with >15 open PRs would drop the
  // tail of the fan-out. Use a tighter 3-attempt (~3s) poll budget here and
  // a wider concurrency window so up to ~100 PRs fit comfortably:
  //   100 PRs / 10 in parallel = 10 batches × ~3s = 30s.
  // PRs whose mergeable is still null after 3s are picked up by the next
  // webhook event (subsequent push, contributor sync, etc.).
  await processInBatches(prs, 10, (pr) =>
    labelMergeConflicts(octokit, pr.number, target, env, /*pollAttempts*/ 3),
  );
}

// GitHub computes `mergeable` lazily and returns null on the first read after
// a relevant change. Poll up to N times (default 5 = ~10s for single-PR
// events; tighter budgets for fan-out paths) before giving up.
async function getMergeableState(octokit, target, prNumber, attempts = 5) {
  const delays = [0, 1000, 2000, 3000, 4000].slice(0, attempts);
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

async function processInBatches(items, batchSize, fn) {
  for (let i = 0; i < items.length; i += batchSize) {
    await Promise.all(items.slice(i, i + batchSize).map(fn));
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
