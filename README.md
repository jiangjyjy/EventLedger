# EventLedger

Reference implementation for typed counterfactual credit assignment over multi-agent orchestration event graphs, including a graph-aware student reward model.

## Repository contents

- `carve/`: event schemas, orchestration, typed counterfactual operators, replay, verifiers, reward construction, and student models.
- `experiments/`: benchmark runners, counterfactual evaluation, calibration, distillation, control, and policy-optimization experiments.
- `tests/`: unit and integration tests using synthetic fixtures.
- `requirements-student.txt`: optional dependencies for Qwen-based student training and evaluation.

Benchmark datasets, API credentials, model weights, trained checkpoints, and experiment outputs are not included. Obtain datasets from their official sources and follow their licenses and access requirements.

## Setup

Use Python 3.10 or newer. Install the base package and optional student dependencies in an isolated environment:

```bash
python -m pip install -e .
python -m pip install -r requirements-student.txt
```

Run the test suite:

```bash
python -m pytest tests -q
```

## API configuration

API-backed experiments expect credentials through environment variables. Set `CARVE_API_KEY` and, for an OpenAI-compatible service, `CARVE_BASE_URLS` to the service's documented API base URL. Multiple fallback URLs may be provided as a comma-separated list. Never commit credentials or populated environment files.

```bash
export CARVE_API_KEY="<your-key>"
export CARVE_BASE_URLS="<openai-compatible-api-base-url>"
```

Use the experiment runner's `--help` output to see dataset and output path options. Paths to datasets, model checkpoints, and output directories should be supplied by the user at runtime rather than embedded in source files.

## Reproducibility notes

The experiments require the matching public benchmark data, a compatible model API for generation and judging, and model weights for local student experiments. Exact commands and split manifests should be recorded with each run. Results from different task splits or evaluation protocols should not be combined as if they were directly comparable.

## Security and privacy

Do not add API keys, SSH configuration, private endpoints, personal filesystem paths, raw run logs, checkpoints, or local datasets to this repository. Before publishing, scan the staged repository and its Git history for secrets and machine-specific metadata.
