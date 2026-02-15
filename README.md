# TiDB Zero Agent Demo

> **Autonomous SQL Agent running on TiDB Zero (Serverless).**
> Spawns a fresh database, fetches real-time data, designs a schema, and writes SQL to answer your questions.

## ⚡ Quick Start

### 1. Install

We recommend [uv](https://github.com/astral-sh/uv) for instant setup:

```bash
# Clone & Sync
git clone https://github.com/lilyjazz/agent-tidb-mvp-demo.git
cd agent-tidb-mvp-demo
uv sync
```

*(Or use standard pip: `python -m venv .venv && source .venv/bin/activate && pip install -e .`)*

### 2. Configure (Choose One)

**Option A: I have a GitHub Copilot / Claude subscription (No API Key needed!)**
If you have the `codex` or `claude` CLI installed and logged in:
```bash
# No .env needed! Just run:
uv run zero-agent-demo run --provider codex_subscription
```

**Option B: I have an API Key (OpenAI / Anthropic / Gemini)**
```bash
cp .env.example .env
# Edit .env and set MODEL_API_KEY=sk-...
```

### 3. Run

**One-Liner (Earthquake Analysis):**
```bash
uv run zero-agent-demo run \
  "Find where magnitude >= 4 earthquakes happened today and summarize clusters." \
  --source-url "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
```

---

## 🍳 Cookbook

Try these live data sources. The agent handles the schema automatically.

**Startup Trends (TechCrunch):**
```bash
uv run zero-agent-demo run \
  "Summarize top 5 startup trends from today's feed." \
  --source-url "https://techcrunch.com/feed/"
```

**Weather Analysis (Tokyo):**
```bash
uv run zero-agent-demo run \
  "Analyze Tokyo's next 48h temp trend. Highlight the biggest drop." \
  --source-url "https://api.open-meteo.com/v1/forecast?latitude=35.68&longitude=139.76&hourly=temperature_2m&forecast_days=2"
```

**Developer Topics (Lobsters):**
```bash
uv run zero-agent-demo run \
  "What are the dominant engineering topics right now?" \
  --source-url "https://lobste.rs/hottest.json"
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
uv run zero-agent-demo replay <run_id>

# See exactly what SQL was executed
uv run zero-agent-demo audit <run_id>

# Get database connection string (to connect manually)
uv run zero-agent-demo conn <run_id>
```

---

## Advanced Configuration

Full list of environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `MODEL_PROVIDER` | `openai`, `anthropic`, `gemini`, `codex_subscription` | `openai` |
| `MODEL_API_KEY` | Required if using API providers | - |
| `MODEL_NAME` | Specific model version (e.g. `gpt-4o`) | Provider default |
| `TIDB_ZERO_TAG` | Tag for the ephemeral instance | `agent-demo` |
