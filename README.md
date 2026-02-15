# TiDB Zero + AutoGen General Agent CLI Demo

This repository is a minimal CLI MVP that demonstrates a **general AutoGen agent** running against **TiDB Cloud Zero**.

It is designed for customer-facing demos where you need to show the full chain clearly: from data fetch to schema decisions, SQL execution, and auditable results.

The demo emphasizes three things in the terminal output:

1. **TiDB Zero provisioning** happens at runtime.
2. The agent **handles schema decisions** from public data.
3. The agent **autonomously writes and executes SQL** to answer a goal.

No fixed business schema or fixed query templates are hardcoded.

## Core Features

- **Per-task isolated TiDB Zero environment**: each `run` provisions a fresh ephemeral TiDB Zero instance.
- **General-agent behavior**: the agent decides whether to reuse existing tables or create new schema based on current data.
- **Autonomous SQL execution**: the agent can create tables, ingest data, and query insights directly in the sandbox DB.
- **Transparent runtime trace**: CLI prints `[TIDB_ZERO]`, `[THINK]`, `[ACTION]`, `[SQL]`, `[OBSERVATION]`, `[FINAL]`.
- **Strong observability**: `replay` restores the full process, `audit` shows SQL trail, `conn` shows connection details for manual verification.
- **Subscription login modes**: supports `codex_subscription` and `claude_subscription` so users can run without entering API keys in project config.

## Business Value for Customers

- **Fast PoC delivery**: show a complete "agent + data" workflow in minutes with public feeds.
- **Trust and explainability**: customers see exactly what the agent did, not just a final answer.
- **Safe experimentation**: isolated disposable TiDB Zero instances reduce blast radius and simplify demos.
- **Governance readiness**: SQL audit trail and replayable runs support internal review and compliance conversations.
- **Reusable sales/solution pattern**: one CLI workflow works across domains (finance, weather, media, dev community, and more).

## What You Need

- Python 3.10+
- One model access method:
  - API key mode (OpenAI / Anthropic / Gemini / OpenAI-compatible), or
  - Subscription mode (`codex_subscription` or `claude_subscription`) with local CLI login

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configure

```bash
cp .env.example .env
```

Fill `.env` with at least:

- `MODEL_PROVIDER`
- `MODEL_NAME`

TiDB Zero invitation code is no longer required.

For this demo, **prefer one-time local login first** (subscription mode, no API key in project env):

```env
# Codex subscription mode
MODEL_PROVIDER=codex_subscription
MODEL_NAME=gpt-5.3-codex
CODEX_SUBSCRIPTION_BIN=codex
```

or

```env
# Claude subscription mode
MODEL_PROVIDER=claude_subscription
MODEL_NAME=sonnet
CLAUDE_SUBSCRIPTION_BIN=claude
```

One-time local login:

```bash
# Codex subscription login
codex login

# Claude subscription login
claude
# then run /login in the interactive session
```

API-key mode is also supported, but treated as secondary for this demo. For OpenAI / Anthropic / Gemini / OpenAI-compatible examples, see comments in `.env.example`.

Optional overrides:

- `MODEL_BASE_URL`
- `MODEL_ORGANIZATION`
- `MODEL_TIMEOUT_SEC` (default `120`)
- `MODEL_MAX_RETRIES` (default `3`)
- `CODEX_SUBSCRIPTION_BIN` (default `codex`)
- `CLAUDE_SUBSCRIPTION_BIN` (default `claude`)
- `TIDB_ZERO_TAG`
- `MAX_TOOL_ITERATIONS`

## Run Cookbook

You can use any public URL. Here are five examples from different domains:

### One-click Codex subscription E2E

If you just want to run the full local E2E flow with minimal setup:

```bash
./run_codex_subscription_e2e.sh
```

Optional variants:

```bash
# Custom goal and source URL
./run_codex_subscription_e2e.sh "Your goal" "https://your-source-url"
```

### Interactive mode (prompt-based)

If you run `run` without arguments, the CLI will prompt you step by step:

```bash
zero-agent-demo run
```

You will then be prompted for:

- goal / question
- source URL

You can also override model settings per run (without editing `.env`):

```bash
zero-agent-demo run --provider anthropic --model claude-3-5-sonnet-latest
```

If model calls are slow in your network, increase timeout for this run:

```bash
zero-agent-demo run --model-timeout-sec 180 --model-max-retries 4
```

### 1) Earthquakes (geoscience)

```bash
zero-agent-demo run \
  "Find where magnitude >= 4 earthquakes are concentrated in the last day and summarize notable clusters." \
  --source-url "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
```

### 2) Weather forecast (climate)

```bash
zero-agent-demo run \
  "Analyze Tokyo's next 48 hours temperature trend and highlight the biggest rise and drop windows." \
  --source-url "https://api.open-meteo.com/v1/forecast?latitude=35.68&longitude=139.76&hourly=temperature_2m&forecast_days=2"
```

### 3) Startup and tech media

```bash
zero-agent-demo run \
  "Summarize the latest TechCrunch stories into top 10 themes, list representative headlines, and choose the most interesting story to deep dive the details." \
  --source-url "https://techcrunch.com/feed/"
```

### 4) Space industry news

```bash
zero-agent-demo run \
  "Summarize recent space industry updates and identify recurring organizations and mission patterns." \
  --source-url "https://api.spaceflightnewsapi.net/v4/articles/?limit=50&format=json"
```

### 5) Developer community trends

```bash
zero-agent-demo run \
  "Analyze current developer-community topics and summarize the most recurring themes and tags." \
  --source-url "https://lobste.rs/hottest.json"
```

Expected CLI tags:

- `[TIDB_ZERO]` provisioning details, expiration, credentials file path
- `[THINK]` agent thought summaries before major actions
- `[ACTION]` non-SQL tool calls (for example `http_fetch`, `schema_inspect`)
- `[SQL]` SQL statements requested by the agent
- `[OBSERVATION]` tool execution outcomes
- `[FINAL]` final answer
- `[AUTONOMY_PROOF]` SQL counts and detected created tables

## Replay and Audit

After a run, you get a `run_id`.

Replay full timeline:

```bash
zero-agent-demo replay <run_id>
```

Inspect SQL audit:

```bash
zero-agent-demo audit <run_id>
```

Show TiDB Zero full connection details (including DB user and password):

```bash
zero-agent-demo conn <run_id>
```

If you need to hide password in terminal output:

```bash
zero-agent-demo conn <run_id> --redact-password
```

## Artifacts

Run artifacts are stored under `.runs/<run_id>/`:

- `timeline.jsonl` human-readable flow lines
- `events.jsonl` raw AutoGen stream event snapshots
- `sql_audit.jsonl` executed SQL trail
- `final_answer.txt` final answer
- `error_traceback.txt` full traceback for failed runs
- `tidb_zero_instance.json` connection credentials (mode `600`)

`tidb_zero_instance.json` contains sensitive credentials. Keep it secure.

## Notes

- TiDB Zero instances are disposable and expire.
- This demo intentionally gives the agent broad SQL freedom inside a dedicated sandbox database (`agent_sandbox` by default).
- The agent is constrained to one SQL statement per tool call for clean auditing.
