# Replay Run Samples

This directory contains successful run snapshots that can be replayed locally.

Included artifacts per run:
- `timeline.jsonl`
- `sql_audit.jsonl`
- `final_answer.txt`
- `perf.json`
- `step_perf.jsonl`

Credentials are intentionally excluded (`tidb_zero_instance.json` is not included).

## Usage

Replay a run:

```bash
RUNS_DIR=examples/replay-runs .venv/bin/zero-agent-demo replay <run_id>
```

Audit SQL and timing:

```bash
RUNS_DIR=examples/replay-runs .venv/bin/zero-agent-demo audit <run_id>
```

## Cases

| Run ID | Prompt | Source URL |
| :--- | :--- | :--- |
| `fe37758b-ba2d-48a8-8d4c-865a41a5a43f` | `What are the dominant engineering topics right now?` | `https://lobste.rs/hottest.json` |
| `42415b9f-e91c-4b0b-87e9-d12604ca7b80` | `Summarize top 1 startup trends from today's feed.` | `https://techcrunch.com/feed/` |
| `934bc942-f0dd-442f-8c48-ff9edd698a9d` | `Analyze Tokyo's next 48h temp trend. Highlight the biggest drop.` | `https://api.open-meteo.com/v1/forecast?latitude=35.68&longitude=139.76&hourly=temperature_2m&forecast_days=2` |
| `7dc4d4f3-c763-4aa7-8fa8-0a2c1b538c15` | `Find and summarize the lastest clusters.` | `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson` |
