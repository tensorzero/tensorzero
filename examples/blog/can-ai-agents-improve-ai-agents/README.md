# Can AI Agents Improve AI Agents? — Starter

Companion code for the blog post [Can AI Agents Improve AI Agents?](https://www.tensorzero.com/blog/can-ai-agents-improve-ai-agents/).

A self-contained starter that lets your own coding agent (Claude Code or Codex) optimize a TensorZero function on the YC Bench Tutorial environment using a local Docker-Compose'd TensorZero gateway, a markdown skill, and a real baseline rollout's traces.

## About this starter

The function you're optimizing is `yc_bench_tutorial_v0::yc_bench_act` — an autonomous CEO agent for the YC Bench business simulation, running on `openai::gpt-5.4-mini`. The metric is `tasks_succeeded` (count of tasks delivered on or before deadline per episode; higher is better). The baseline `initial` variant scores **0.800** on the 20-episode test split.

The starter ships the function's `run_command` tool schema (so the gateway accepts well-formed inference requests), but **the YC Bench simulator itself is not installed locally**. Probes you fire from this starter validate prompt structure and tool-call shape — not actual `yc-bench` execution. End-to-end evaluation against the simulator is the user's job after the agent exits.

The included baseline traces come from a Codex YC Bench seed 0 rollout; nothing in the skill or methodology is Codex-specific, and Claude Code uses the exact same files.

## Prerequisites

- Docker (with Docker Compose v2)
- An `OPENAI_API_KEY`
- A coding agent: [Claude Code](https://claude.com/claude-code) or [Codex](https://github.com/openai/codex) on your `$PATH` (or the desktop app)

## Quick start

### From the blog button (Open in Claude Code)

If you arrived here from the _Open in Claude Code_ button on the blog post, Claude Code is already running with a single message that points it at `SKILL.md`. The agent will run through the Setup section of `SKILL.md` itself — clone the repo if needed, bring up the gateway, fetch the baseline data — and then proceed to the optimization loop. You only need to make sure `$OPENAI_API_KEY` is set in your shell before the agent's first `docker compose` step runs. (If you didn't arrive that way, follow the manual quick-start below.)

### Quick start (manual)

```bash
git clone https://github.com/tensorzero/tensorzero
cd tensorzero/examples/blog/can-ai-agents-improve-ai-agents

# Bring up the stack (gateway + Postgres; Postgres is needed for experimentation routing).
# A one-shot migrations service runs automatically before the gateway starts.
export OPENAI_API_KEY=sk-...
docker compose up -d
curl -sf http://localhost:3000/status   # should print {"gateway":"ok",...}

# Pull the baseline rollout traces (~98 MiB; hosted via Git LFS in a sibling data repo):
bash baseline_data/fetch.sh

# Smoke-test the gateway against the initial variant:
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

# Then launch your coding agent in this directory:
claude   # or: codex
# Paste the contents of SKILL.md as the first message.
```

Stop the gateway when you're done: `docker compose down`.

## How it works

The agent reads `SKILL.md` and follows a survey → diagnose → write → restart → probe → iterate loop:

1. Surveys `baseline_data/inferences.jsonl` and `feedback.jsonl` to find recurring failure modes in the baseline rollout.
2. Writes 1–3 new prompt variants under `functions/yc_bench_tutorial_v0__yc_bench_act/<variant_name>/`.
3. Registers them in `tensorzero.toml`.
4. Restarts the gateway: `docker compose restart gateway`. (The gateway loads its config once at startup, so config edits require a restart.)
5. Probes each variant with `curl -X POST http://localhost:3000/inference -d ...`.
6. When done, the agent leaves an `[experimentation]` block in `tensorzero.toml` listing its best candidates and exits. **Running the full evaluation is the user's job after the agent exits.**

The compose file mounts this directory into the gateway container at `/app/config` read-write, so the agent's edits to `tensorzero.toml` / `functions/` / `tools/` flow through to the container after each restart.

## What you've got

- `docker-compose.yml` — three-service stack: the gateway (port 3000), Postgres (port 5432, persistent volume `postgres-data`, image `tensorzero/postgres:17` which ships pg_cron + pgvector), and a one-shot migrations service that applies Postgres migrations on first start. Postgres is required because the experimentation routing the agent uses (`track_and_stop` / adaptive) writes per-variant trial counts and means.
- `tensorzero.toml` — starting config. Defines the function, metrics, the `run_command` tool, and a single `initial` variant on `gpt-5.4-mini`.
- `functions/yc_bench_tutorial_v0__yc_bench_act/` — the `initial` variant's templates (`system_template.minijinja`, `user_template.minijinja`) and `user_schema.json`.
- `tools/run_command.json` — the tool schema (the only tool the YC Bench agent uses).
- `baseline_data/fetch.sh` — downloads the rollout traces (hosted via Git LFS in [anndvision/data](https://github.com/anndvision/data/tree/main/can-ai-agents-improve-ai-agents/baseline_data)) into this directory and verifies their SHA-256 checksums.
- `baseline_data/inferences.jsonl` (~98 MiB, 1,380 rows; gitignored, fetched by `fetch.sh`) — every inference from a real baseline rollout (Codex arm, YC Bench seed 0; 80 train + 20 test episodes against the initial variant).
- `baseline_data/feedback.jsonl` (gitignored, fetched by `fetch.sh`) — every metric value from that same rollout, keyed by `target_id` (matches `episode_id` for episode-level metrics).
- `baseline_data/initial_config/` — frozen copy of the starting tensorzero.toml + functions tree, so the agent has a read-only reference even after editing the live config.
- `SKILL.md` — the agent's playbook: how to read the data, what failure modes to look for, the canonical inference-request shape, the restart-after-edit step, and the experimentation-block template.

## Troubleshooting

- **`docker compose up -d` errors on `OPENAI_API_KEY must be set`** — `export OPENAI_API_KEY=sk-...` in the same shell before bringing the stack up.
- **`curl http://localhost:3000/status` hangs / refuses connection** — gateway hasn't started yet. `docker compose ps` to check status, `docker compose logs gateway` to read its output.
- **Gateway logs mention "no postgres connection" / migration errors** — the gateway expects Postgres on `postgres://postgres:postgres@postgres:5432/tensorzero` (set in `docker-compose.yml`). The one-shot `gateway-run-postgres-migrations` service should apply migrations on first `docker compose up -d`; if it didn't, check `docker compose logs gateway-run-postgres-migrations`. Postgres data persists in the `postgres-data` volume — `docker compose down -v` wipes it and the next `up -d` will re-migrate.
- **Gateway logs mention `pg_cron extension is not installed`** — you're using a vanilla `postgres:*` image instead of `tensorzero/postgres:17`. Make sure `docker-compose.yml`'s `postgres` service still uses `tensorzero/postgres:17` and starts with `command: ["postgres", "-c", "cron.database_name=tensorzero"]`.
- **Frozen vs. live config** — `baseline_data/initial_config/tensorzero.toml` is a read-only reference to the starter's day-one config and should match the live `./tensorzero.toml` unless you intentionally want to revert.
- **Agent's edits to `tensorzero.toml` aren't taking effect** — the gateway reads its config once at startup. Run `docker compose restart gateway` and wait for `curl -sf http://localhost:3000/status` to return ok before probing.
- **Agent gets stuck on tool-call shape** — point it at `SKILL.md` § _Templates, schemas, and the required `content` shape_. The function is **legacy-style** (per-role schemas), not the newer named-template style.
- **The "Open in Claude Code" button does nothing.** Claude Code's `claude://` URL scheme requires the desktop app (or a CLI handler) to be installed and registered as the protocol handler. If your browser shows "no application found" or just silently fails: install Claude Code first, or fall back to the manual quick-start above.
- **The agent can't find SKILL.md / clones into a weird directory.** The deeplink prompt asks the agent to fetch `SKILL.md` from the raw.githubusercontent.com URL and follow it. If your agent's session started in `~/Downloads` or somewhere unexpected, the clone will land there; cancel the agent, `cd` somewhere sensible, and paste the prompt again. (Or: tell the agent in plain English where you'd like the repo to live before it clones.)
