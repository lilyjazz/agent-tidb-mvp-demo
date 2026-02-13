# TiDB Zero + AutoGen General Agent CLI Demo

[![Launch TiDB Cloud Zero](https://img.shields.io/badge/Launch-TiDB%20Cloud%20Zero-00C1DE?style=for-the-badge)](https://zero.tidbcloud.com/)

> [!IMPORTANT]
> Start with **TiDB Cloud Zero**: **[https://zero.tidbcloud.com/](https://zero.tidbcloud.com/)**

**Agent-native data analysis, without prebuilding a schema.**

This repository is a minimal CLI MVP that demonstrates a **general AutoGen agent** running against **TiDB Cloud Zero**. The agent provisions a TiDB Zero instance at runtime, ingests public data, makes schema decisions, and executes SQL to answer your goal.

## Why TiDB Zero stands out in this demo

- **Runtime provisioning**: spin up the database while the agent runs.
- **Disposable environments**: ideal for fast experiments; instances expire automatically.
- **Isolated sandbox**: the agent operates in a dedicated database (`agent_sandbox` by default).
- **Auditable autonomy**: one SQL statement per tool call with a full SQL trail.
- **Schema-flexible workflow**: no fixed business schema or query templates are hardcoded.

## What You Need

- Python 3.10+
- A cloud model key (`OPENAI_API_KEY`)
- A TiDB Zero invitation code (`TIDB_ZERO_INVITATION_CODE`)

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

- `OPENAI_API_KEY`
- `TIDB_ZERO_INVITATION_CODE`

Optional overrides:

- `MODEL_NAME` (default: `gpt-4o-mini`)
- `TIDB_ZERO_TAG`
- `MAX_TOOL_ITERATIONS`

## Run Cookbook

You can use any public URL. Here are five examples from different domains:

### Interactive mode (prompt-based)

If you run `run` without arguments, the CLI will prompt you step by step:

```bash
zero-agent-demo run
```

You will then be prompted for:

- goal / question
- source URL

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
  "Summarize the latest TechCrunch stories into top 10 themes and list representative headlines." \
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
- `[THINK]` model reasoning messages
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

## Artifacts

Run artifacts are stored under `.runs/<run_id>/`:

- `timeline.jsonl` human-readable flow lines
- `events.jsonl` raw AutoGen stream event snapshots
- `sql_audit.jsonl` executed SQL trail
- `final_answer.txt` final answer
- `tidb_zero_instance.json` connection credentials (mode `600`)

`tidb_zero_instance.json` contains sensitive credentials. Keep it secure.

## Notes

- TiDB Zero instances are disposable and expire.
- This demo intentionally gives the agent broad SQL freedom inside a dedicated sandbox database (`agent_sandbox` by default).
- The agent is constrained to one SQL statement per tool call for clean auditing.
