# TiDB Zero Agent Demo

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TiDB Zero](https://img.shields.io/badge/TiDB-Serverless-ff0066)](https://pingcap.com/ai)

> **Autonomous SQL Agent.** Spawns a dedicated serverless database, ingests data from any URL, designs a schema, and writes SQL to answer your questions—all in seconds.

---

## 🚀 Features

- **Zero-Config Infrastructure**: Automatically provisions ephemeral TiDB Zero (Serverless) instances.
- **Autonomous Data Engineering**: Analyzes raw data (JSON/CSV/XML), designs normalized schemas, and handles ingestion.
- **Self-Healing SQL**: Writes, executes, and fixes SQL queries automatically based on execution errors.
- **Multi-Provider Support**: Works with OpenAI, Anthropic, Gemini, or local CLI subscriptions (Codex/Claude).
- **Full Auditability**: Every thought, SQL query, and result is logged for replay and audit.

## ⚡ Quick Start

### 1. Installation

Requires Python 3.10+.

```bash
git clone https://github.com/lilyjazz/agent-tidb-mvp-demo.git
cd agent-tidb-mvp-demo

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configuration

Choose your preferred AI provider:

#### Option A: API Key (Recommended)
Create a `.env` file:

```bash
cp .env.example .env
```
Edit `.env` to set your API key (e.g., `MODEL_API_KEY=sk-...` for OpenAI/Anthropic/Gemini).

#### Option B: CLI Subscription (No API Key)
If you have `codex` or `claude` CLI tools installed and authenticated, no `.env` is required. Just use the `--provider` flag.

### 3. Usage

**Analyze Tokyo Weather:**
```bash
zero-agent-demo run \
  "Analyze Tokyo's next 48h temp trend. Highlight the biggest drop." \
  --source-url "https://api.open-meteo.com/v1/forecast?latitude=35.68&longitude=139.76&hourly=temperature_2m&forecast_days=2"
```

**Analyze Tech Trends:**
```bash
zero-agent-demo run \
  "Summarize top startup trends from today's feed." \
  --source-url "https://techcrunch.com/feed/"
```

**Custom Provider:**
```bash
zero-agent-demo run "Analyze this data..." --source-url "..." --provider anthropic
```

## 🔍 Observability

Every run produces a detailed artifact trail in `.runs/`.

| Command | Description |
| :--- | :--- |
| `zero-agent-demo replay <run_id>` | Replay the agent's thought process step-by-step. |
| `zero-agent-demo audit <run_id>` | View all executed SQL statements and their results. |
| `zero-agent-demo conn <run_id>` | Get the connection string to connect to the ephemeral DB manually. |

### Example Replays
We include sample runs in `examples/replay-runs`:

```bash
# Replay a pre-recorded run analyzing Hacker News trends
RUNS_DIR=examples/replay-runs zero-agent-demo replay fe37758b-ba2d-48a8-8d4c-865a41a5a43f
```

## 🛠 Supported Providers

| Provider | CLI Flag | Auth | Status |
| :--- | :--- | :--- | :--- |
| **OpenAI** | `openai` | `MODEL_API_KEY` | ✅ Stable |
| **Codex CLI** | `codex_subscription` | `codex login` | ✅ Stable |
| **Claude CLI** | `claude_subscription` | `claude login` | ✅ Stable |
| **Anthropic** | `anthropic` | `MODEL_API_KEY` | ⚠️ Beta |
| **Gemini** | `gemini` | `MODEL_API_KEY` | ⚠️ Beta |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
