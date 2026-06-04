# TensorZero Function Optimizer

You are optimizing a TensorZero function to improve its performance metric.

## Setup (run this before anything else)

You may have arrived here in one of two ways:

- **Via the blog's "Open in Claude Code" deeplink** — your cwd is wherever the user happened to be (probably `~`, `~/Downloads`, or similar), and nothing about this repo is set up yet.
- **Manually** — the user already cloned the repo, ran `claude` (or `codex`) from inside `examples/blog/can-ai-agents-improve-ai-agents/`, and pasted these instructions. In that case most of the steps below are no-ops; check and skip.

Walk through these in order. Each step is idempotent — re-running them when they're already done is safe. **Do not run a full evaluation as part of setup**; setup ends at the smoke-test curl. Real evaluation episodes are the user's job after you exit (see Methodology below).

### 0. Open with a friendly summary, then get started

Before running any commands, print a short, plain-English summary so the user knows what's about to happen. Two short paragraphs is plenty — keep it conversational, light on jargon. After the summary, just start with step 1; don't ask for permission.

**Paragraph one.** In a couple of sentences:

- We're going to improve an AI agent that plays the role of a startup CEO in a business simulation. The agent has one tool — the simulator's command-line — and a goal: deliver as many tasks as possible before their deadlines.
- It currently uses GPT-5.4-mini with a generic prompt and finishes about 0.8 tasks per run. The plan is to analyze historical traces and feedback, identify where it's failing, and rewrite the prompt accordingly. You'll be able to deploy adaptive A/B tests with the new prompts after I'm done.

**Paragraph two — what I'll do next.** Three or four conversational bullets is enough:

- Get the example code on your machine (or pick up where you left off if it's already there).
- Spin up a local stack with Docker so the agent has somewhere to run.
- Grab the recorded traces of the baseline runs (~98 MB, takes a few seconds).
- Fire one quick test inference to make sure your OpenAI key works end-to-end. This costs a few cents.

Tell the user what you'll need from them inline as you go (e.g. an `OPENAI_API_KEY` if it isn't already wired up), but don't pre-emptively block on it — start with the steps that don't need it (clone, Docker pull) so you're ready to go the moment they paste the key.

If you discover something that genuinely requires a user decision — a pre-existing checkout with uncommitted work, an already-edited `tensorzero.toml`, multiple plausible places to clone into — flag it briefly and ask, then continue. Otherwise just press on.

### 1. Locate (or clone) the repo and `cd` into the example directory

```bash
# Already in the right place?
case "$(pwd)" in
  */examples/blog/can-ai-agents-improve-ai-agents) echo "Already in example dir."; ;;
  *)
    # Inside the tensorzero repo but elsewhere? cd from the repo root.
    if root=$(git rev-parse --show-toplevel 2>/dev/null) \
       && [ -d "$root/examples/blog/can-ai-agents-improve-ai-agents" ]; then
      cd "$root/examples/blog/can-ai-agents-improve-ai-agents"
    # Already cloned somewhere under cwd?
    elif [ -d ./tensorzero/examples/blog/can-ai-agents-improve-ai-agents ]; then
      cd ./tensorzero/examples/blog/can-ai-agents-improve-ai-agents
    # Otherwise clone fresh.
    else
      git clone https://github.com/tensorzero/tensorzero
      cd tensorzero/examples/blog/can-ai-agents-improve-ai-agents
    fi
    ;;
esac
pwd
```

If `git clone` fails because Docker / git / curl aren't installed, stop and ask the user to install the missing tool before continuing.

If the clone already exists but is at an old revision (`git rev-parse HEAD` differs from `origin/main`), warn the user and ask whether to `git pull` — don't pull on your own, since they may have in-progress edits.

### 2. Make `OPENAI_API_KEY` available to the gateway

You (the agent) will be running `docker compose up -d`, not the user. The key only needs to reach the gateway container — it does **not** need to be in your shell. Check whether either of these is already true (in priority order); if neither, ask the user for the key and walk them through Option A.

- A `.env` file already exists next to `docker-compose.yml` containing `OPENAI_API_KEY=...`. Docker Compose auto-loads it.
- `OPENAI_API_KEY` is already exported in the shell you'll run `docker compose` from.

Quick check:

```bash
{ [ -f .env ] && grep -q '^OPENAI_API_KEY=' .env; } || [ -n "${OPENAI_API_KEY:-}" ] \
  && echo "key is available" \
  || echo "no key yet"
```

If "no key yet": stop and ask the user. Recommend, in order of preference:

**Option A — local `.env` file (simplest, what most users want):**

```bash
# Create the .env file next to docker-compose.yml. Don't echo or print the key.
printf 'OPENAI_API_KEY=sk-...\n' > .env
chmod 600 .env
```

The file is gitignored at the repo root, only readable by the user, and Docker Compose picks it up automatically — you don't need to export anything in your shell. Tell the user to swap `sk-...` for their real key in their own terminal (don't ask them to paste the key into the chat).

**Option B — macOS Keychain (no plaintext on disk):**

```bash
# One-time, run by the user:
security add-generic-password -a "$USER" -s openai-api-key -w

# Then per `docker compose` invocation:
OPENAI_API_KEY=$(security find-generic-password -w -s openai-api-key) docker compose up -d
```

The key is briefly visible in `ps` while the command runs but never written to disk. If the user already has a system credential manager (1Password CLI's `op read`, `pass`, etc.), the same shape works.

**Option C — same-shell export:** `export OPENAI_API_KEY=sk-...` in the terminal you'll launch `docker compose` from. Simple but writes the key to shell history; less safe than A or B.

Things to avoid regardless of which option: don't print the key to the terminal, don't paste it into the chat, don't put it in `tensorzero.toml` or `docker-compose.yml`, and don't commit any file containing it.

Don't proceed to step 3 without a key in place — the gateway will start fine, but the smoke test will fail with a confusing 401.

### 3. Bring up the gateway

The compose stack has three services: `postgres` (metadata store — required for `track_and_stop` / adaptive experimentation routing), a one-shot `gateway-run-postgres-migrations` service that applies migrations and exits, and the `gateway` itself. `docker compose up -d` orchestrates all three in the right order:

```bash
docker compose up -d
```

If this fails, read the error before retrying. The common modes and what to recommend:

- **`Cannot connect to the Docker daemon` / `error during connect`** — Docker is installed but not running. Recommend, by OS:
  - **macOS / Windows:** open the Docker Desktop application (Spotlight / Start menu) and wait for the whale icon to settle. Then re-run `docker compose up -d`. On macOS the agent can also try `open -a Docker` from the shell. Don't recommend running Docker as root.
  - **Linux:** `sudo systemctl start docker` (systemd) or `sudo service docker start`. The user may also need to be in the `docker` group (`sudo usermod -aG docker $USER`, then log out and back in) — recommend that over prefixing every command with `sudo`. Avoid `--privileged` flags.
- **`docker: command not found`** — Docker isn't installed. Recommend Docker Desktop from <https://www.docker.com/products/docker-desktop/> on macOS / Windows, or the official package for the user's Linux distro (e.g. `apt install docker.io` on Debian/Ubuntu). Re-run after installing.
- **`Bind for 0.0.0.0:3000 failed: port is already allocated`** — something else is on port 3000. Surface the message verbatim. Suggest the user run `lsof -iTCP:3000 -sTCP:LISTEN` (or `ss -ltnp 'sport = :3000'` on Linux) to find the process, stop it, then retry. Don't quietly remap the port — `tensorzero.toml` and the curl smoke test below assume 3000.
- **Anything else** — surface the error verbatim and stop; don't guess.

Then poll for readiness (~30s max):

```bash
for i in $(seq 1 30); do
  if curl -sf http://localhost:3000/status >/dev/null; then
    echo "Gateway ready."; break
  fi
  sleep 1
done
curl -sf http://localhost:3000/status || { docker compose logs gateway | tail -50; echo "Gateway didn't come up — read the logs above."; exit 1; }
```

If the healthcheck never goes ok, dump `docker compose logs gateway` and ask the user to read it before continuing.

### 4. Fetch the baseline traces

```bash
bash baseline_data/fetch.sh
```

Idempotent: the script SHA-256-verifies and skips files that are already present.

### 5. Smoke-test the `initial` variant

One inference call confirms the full path (Docker → gateway → OpenAI → response) before iterating.

```bash
curl -sf -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "yc_bench_tutorial_v0::yc_bench_act",
    "variant_name": "initial",
    "input": {
      "messages": [{
        "role": "user",
        "content": [{
          "type": "template",
          "name": "user",
          "arguments": { "observation": "## Simulation Start — funds: $250,000, payroll: $14,944/mo, employees: 3, prestige: 1." }
        }]
      }]
    }
  }' | python -m json.tool
```

You should see a non-empty assistant message — typically a `tool_call` with `name: "run_command"`. If you instead see a 401 (`Incorrect API key`), the key isn't set or is wrong; go back to step 2. Other errors should be surfaced verbatim and stopped on.

### 6. Print a clear "ready" handoff

Tell the user something like: "Setup complete. Gateway at http://localhost:3000 with the `initial` variant working. Starting baseline analysis." This makes the boundary between setup and the real optimization work obvious.

Then continue with the methodology below.

## Environment

- T0 config files: `./` (only these and the baseline data below are relevant — don't explore elsewhere).
- Gateway URL: `http://localhost:3000` (a Docker Compose stack defined in `./docker-compose.yml` — gateway + Postgres + a one-shot migrations service).
- Pre-dumped baseline data: `./baseline_data/` (read-only; direct DB access is not available).
- After editing `tensorzero.toml` or any file under `functions/` / `tools/`, restart the gateway: `docker compose restart gateway` then re-check `curl -sf http://localhost:3000/status`.
- Observability uses Postgres as the backend (`[gateway.observability] backend = "postgres"` in `tensorzero.toml`); this is required so the experimentation routing block (`track_and_stop` / adaptive) works. ClickHouse and Valkey are not used by this starter.
- Use whatever tooling the host has (`curl`, `grep`, `sort`, `head`, plus typically some combination of `node`, `python`, `jq`, `awk`) — pick what's available and ergonomic for the projection at hand. The examples below use `node -e` because that's what the eval harness used; substitute `python -c`, `jq`, etc. if you prefer.
- Don't set `temperature` on any variant (some models reject non-default values). Keep the `initial` variant in place as a baseline reference.
- Don't run a full evaluation yourself — that's the user's job after you exit.

## Task

- Function: `yc_bench_tutorial_v0::yc_bench_act`
- Metric: `tasks_succeeded` (float, level = `episode`, optimize = `max`). It's an integer count of YC Bench tasks delivered on or before their deadlines per episode. Higher is better.
- Baseline performance (initial variant on `openai::gpt-5.4-mini`):
  - test (n = 20): `tasks_succeeded` mean = 0.800
  - train (n = 80): `tasks_succeeded` mean = 1.150

## Available Models

- `openai::gpt-5.4-mini` (only)

## Baseline data

- `baseline_data/inferences.jsonl` — one row per inference (what the model said per turn).
- `baseline_data/feedback.jsonl` — one row per metric value, keyed by `target_id` (which matches `episode_id` for episode-level metrics like `tasks_succeeded`).
- `baseline_data/initial_config/` — read-only copy of the starting T0 config tree.

These were fetched in Setup above. If for any reason the files aren't there (e.g. you skipped Setup), re-run `bash baseline_data/fetch.sh` — it's idempotent and SHA-verifies.

`inferences.jsonl` is ~98 MiB. Don't `cat` it whole. Start by `head -3` on each file to learn the row shape, then project out only the fields you need.

### The projection pattern

`grep` to narrow, then a small one-liner to project. Example with `node`:

```bash
grep "$EPISODE_ID" baseline_data/inferences.jsonl \
  | node -e "
      require('readline').createInterface({input: process.stdin}).on('line', l => {
        const r = JSON.parse(l);
        console.log(r.id, r.variant_name, JSON.stringify(r.output).slice(0,200));
      });"
```

`cat inferences.jsonl | ...` loads the whole file; `grep`-first keeps the pipeline cheap.

### Cross-record one-liners

```bash
# How many inferences per episode (most-active first)
grep -o '"episode_id":"[^"]*"' baseline_data/inferences.jsonl | sort | uniq -c | sort -rn | head

# Last inference of a specific episode
grep "$EPISODE_ID" baseline_data/inferences.jsonl | tail -1

# Which metrics are present and how many of each
grep -o '"metric_name":"[^"]*"' baseline_data/feedback.jsonl | sort | uniq -c

# tasks_succeeded values per episode (sorted ascending — bad → good)
grep '"metric_name":"tasks_succeeded"' baseline_data/feedback.jsonl \
  | node -e "
      const lines = [];
      require('readline').createInterface({input: process.stdin})
        .on('line', l => lines.push(JSON.parse(l)))
        .on('close', () => {
          lines.sort((a, b) => a.value - b.value);
          for (const r of lines) console.log(r.target_id, r.value);
        });"

# Episode IDs of the worst-performing test episodes (value == 0)
grep '"metric_name":"tasks_succeeded"' baseline_data/feedback.jsonl \
  | node -e "
      require('readline').createInterface({input: process.stdin}).on('line', l => {
        const r = JSON.parse(l);
        if (r.value === 0) console.log(r.target_id);
      });" > /tmp/zero_episodes.txt
head -5 /tmp/zero_episodes.txt | while read id; do
    echo "=== $id ==="
    grep "$id" baseline_data/inferences.jsonl | head -1
done
```

### Templates, schemas, and the required `content` shape

This function uses the **legacy** per-role config style. The function **name** in the TOML keys keeps `::` because that's the actual function identifier; the on-disk **directory** uses `__` instead so the path stays Windows-safe (Windows filesystems reject `:`). When you create a new variant, follow the same convention: `functions/yc_bench_tutorial_v0__yc_bench_act/<variant_name>/...`. From `tensorzero.toml`:

```toml
[functions."yc_bench_tutorial_v0::yc_bench_act"]
type = "chat"
user_schema = "functions/yc_bench_tutorial_v0__yc_bench_act/user_schema.json"
tools = ["run_command"]

[functions."yc_bench_tutorial_v0::yc_bench_act".variants.initial]
type = "chat_completion"
model = "openai::gpt-5.4-mini"
system_template = "functions/yc_bench_tutorial_v0__yc_bench_act/initial/system_template.minijinja"
user_template = "functions/yc_bench_tutorial_v0__yc_bench_act/initial/user_template.minijinja"
```

The user schema requires a single `observation` string. The canonical `content` block for an inference request:

```json
"content": [{
  "type": "template",
  "name": "user",
  "arguments": { "observation": "..." }
}]
```

For a role with no schema (e.g. assistant): `"content": "..."` or `[{"type": "text", "text": "..."}]`.

### Probing a variant

```bash
curl -sf -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "yc_bench_tutorial_v0::yc_bench_act",
    "variant_name": "your_new_variant",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": [{
            "type": "template",
            "name": "user",
            "arguments": { "observation": "## Simulation Start — funds: $250,000, payroll: $14,944/mo, employees: 3, prestige: 1." }
          }]
        }
      ]
    }
  }' | python -m json.tool
```

For multi-turn agentic envs like YC Bench, a single turn-0 probe tells you very little. Pull a real episode out of `inferences.jsonl`, copy its first 2–3 messages into your curl body, and check how the new variant continues. Look for: right tool call (`run_command`), reasonable command, sensible step-by-step plan.

## Methodology

The core loop is: survey the baseline → diagnose recurring failure modes → write 1–3 prompt variants → register them in `tensorzero.toml` → restart the gateway → probe → iterate → exit.

Decisions worth getting right:

- **Read the metric direction.** `tasks_succeeded` maximizes — high values are better. The worst test episodes scored 0 (no tasks delivered on time); the best scored 4–5.
- **Diagnose from real failures.** Pull the worst test episodes (`value == 0`), read their last few inferences end-to-end, and form your own hypotheses about what went wrong. Resist generic guesses — the failure modes you find should be specific to what the trace actually shows.
- **Write new templates as files**, not inline. Each new variant gets its own directory under `functions/yc_bench_tutorial_v0__yc_bench_act/<variant_name>/` with a `system_template.minijinja` (and `user_template.minijinja` if you change it; otherwise reuse `initial/user_template.minijinja`).
- **Register variants in `tensorzero.toml`.** Add a `[functions."yc_bench_tutorial_v0::yc_bench_act".variants.<variant_name>]` block per new variant — same shape as `initial`, just different `model` / `system_template` paths.
- **Restart the gateway after each config edit.** The gateway loads `tensorzero.toml` once at startup. After any change to the config or templates: `docker compose restart gateway`, wait for `curl -sf http://localhost:3000/status` to return ok, then probe.
- **Judge probes by the curl response itself** — does the assistant call `run_command` with a sensible argument? Does it follow the strategy laid out in the new prompt? It's fine if a probe doesn't directly improve `tasks_succeeded` (that requires a real episode rollout, which the user runs after you exit) — the goal is variant designs that are plausibly better than `initial`.
- **When done, leave the best variant(s) in place** with an experimentation block (see below) and exit. Don't run the full evaluation yourself — the user does that.

## Routing: Experimentation Config

After creating new variants, add an experimentation section to `tensorzero.toml` so the gateway concentrates evaluation episodes on your best candidates rather than round-robining everything. Keep candidates to ~3–4, including `initial` as a baseline.

```toml
[functions."yc_bench_tutorial_v0::yc_bench_act".experimentation]
type = "track_and_stop"
metric = "tasks_succeeded"
candidate_variants = ["initial", "your_new_variant_1", "your_new_variant_2"]
fallback_variants = []
min_samples_per_variant = 5
delta = 0.1
epsilon = 0.0
update_period_s = 5
min_prob = 0.0
max_samples_per_variant = 10000
```
