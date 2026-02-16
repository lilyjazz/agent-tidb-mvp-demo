# TiDB Zero Agent Demo

> **Autonomous SQL Agent running on TiDB Zero (Serverless).**
> Spawns a fresh database, fetches real-time data, designs a schema, and writes SQL to answer your questions.

## ⚡ Quick Start

### 1. Install

```bash
git clone https://github.com/lilyjazz/agent-tidb-mvp-demo.git
cd agent-tidb-mvp-demo
python3 -m venv .venv
.venv/bin/pip install -e .
```

### 2. Configure (Choose One)

#### Option A: I have a Codex / Claude subscription (No API Key needed! :sparkles:)
If you have the `codex` or `claude` CLI installed and logged in.

No .env needed! Just run the step 3 :rocket:

#### Option B: I have an API Key (OpenAI / Anthropic / Gemini) :hammer:
```bash
cp .env.example .env
# Edit .env and set MODEL_API_KEY=sk-...
```

### 3. Run

**:flight_departure: One-Liner (Tokyo Weather):**
```bash
.venv/bin/zero-agent-demo run \
  "Analyze Tokyo's next 48h temp trend. Highlight the biggest drop." \
  --source-url "https://api.open-meteo.com/v1/forecast?latitude=35.68&longitude=139.76&hourly=temperature_2m&forecast_days=2"
```

### 4. Support Status (Prioritize Tested Paths)

The following is the current validation status in this repo:

| Method | `MODEL_PROVIDER` | Auth | Status | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| OpenAI API | `openai` | API Key | ✅ Tested | Use first |
| Codex Subscription | `codex_subscription` | Codex CLI login | ✅ Tested | Use first |
| Claude Subscription | `claude_subscription` | Claude CLI login | ⚠️ Not fully tested | Use after tested paths |
| Anthropic API | `anthropic` | API Key | ⚠️ Not fully tested | Use after tested paths |
| Gemini API | `gemini` | API Key | ⚠️ Not fully tested | Use after tested paths |
| OpenAI-compatible API | `openai_compatible` | API Key + Base URL | ⚠️ Not fully tested | Use after tested paths |

---

## 🍳 Cookbook

Try these live data sources. The agent handles the schema automatically.

**Startup Trends (TechCrunch):**
```bash
.venv/bin/zero-agent-demo run \
  "Summarize top 1 startup trends from today's feed." \
  --source-url "https://techcrunch.com/feed/"
```

**Developer Topics (Lobsters):**
```bash
.venv/bin/zero-agent-demo run \
  "What are the dominant engineering topics right now?" \
  --source-url "https://lobste.rs/hottest.json"
```

**Earthquake Analysis (USGS):**
```bash
.venv/bin/zero-agent-demo run \
  "Find and summarize the latest clusters." \
  --source-url "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
```

---

## 🛠 How It Works

1.  **Provision:** CLI requests a fresh, ephemeral TiDB Zero instance (no signup needed).
2.  **Fetch:** Agent grabs data from your URL (JSON/XML/CSV).
3.  **Design:** Agent analyzes data structure and `CREATE TABLE`.
4.  **Ingest:** Agent inserts data into TiDB.
5.  **Analyze:** Agent writes SQL queries to answer your question.
6.  **Cleanup:** Database expires automatically.

## 🔍 Audit & Replay

Every run is recorded in `.runs/`.

```bash
# Replay the thought process
.venv/bin/zero-agent-demo replay <run_id>

# See exactly what SQL was executed
.venv/bin/zero-agent-demo audit <run_id>

# Get database connection string (to connect manually)
.venv/bin/zero-agent-demo conn <run_id>
```

Sample successful replay snapshots are available under `examples/replay-runs/`.

```bash
RUNS_DIR=examples/replay-runs .venv/bin/zero-agent-demo replay <run_id>
RUNS_DIR=examples/replay-runs .venv/bin/zero-agent-demo audit <run_id>
```

Preloaded replay run IDs and scenarios:

| Run ID | Scenario | Source |
| :--- | :--- | :--- |
| `fe37758b-ba2d-48a8-8d4c-865a41a5a43f` | Dominant engineering topics from Lobsters hottest feed | `https://lobste.rs/hottest.json` |
| `42415b9f-e91c-4b0b-87e9-d12604ca7b80` | Top startup trend from today's TechCrunch feed | `https://techcrunch.com/feed/` |
| `934bc942-f0dd-442f-8c48-ff9edd698a9d` | Tokyo weather next 48h trend and biggest drop | `https://api.open-meteo.com/v1/forecast?latitude=35.68&longitude=139.76&hourly=temperature_2m&forecast_days=2` |
| `7dc4d4f3-c763-4aa7-8fa8-0a2c1b538c15` | Earthquake cluster summary from USGS all-day feed | `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson` |

---

## Advanced Configuration

Full list of environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `MODEL_PROVIDER` | `openai`, `anthropic`, `gemini`, `openai_compatible`, `claude_subscription`, `codex_subscription` | `openai` |
| `MODEL_API_KEY` | Required if using API providers | - |
| `MODEL_NAME` | Specific model version (e.g. `gpt-4o`) | Provider default |
| `TIDB_ZERO_TAG` | Tag for the ephemeral instance | `agent-demo` |
| `BATCH_TOOLS (EXPERIMENTS)` | Enable multi-action decisions in subscription mode (`true`/`false`) | `false` |
