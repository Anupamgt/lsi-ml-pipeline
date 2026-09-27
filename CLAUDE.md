# CLAUDE.md — lsi-ml-pipeline

Landslide Susceptibility Index (LSI) pipeline for Aizawl, Mizoram: 17-factor Frequency Ratio + RF/LR with buffered spatial LOOCV (1 km). IIT Patna B.Tech project.

## Current work: Vision-Language (VLM) extension
**Before any VLM task, read `VLM_KICKOFF.md` in full.** It holds the plan, the repo state, known data issues, and the phase checkpoints. Work one phase at a time and stop at each checkpoint for review.

## Non-negotiables
- Aizawl is the **test set only**. It never goes into VLM training data (exclude the Aizawl extent + 5 km).
- The VLM **complements** FR/ML. Don't change the behaviour of `lsi_pipeline/` without asking.
- Run `git status` before branching or committing. Never force-push, and never push without asking.
- API keys come from env vars only. Never write keys to files.
- Windows machine: use `pathlib`, Python ≥ 3.11.
- New VLM code goes in `lsi_vlm/`, outputs go in `outputs_vlm/` (gitignored), and optional deps go in the `vlm` extra of `pyproject.toml`.

## Useful commands
- Install: `pip install -e .[dev]`
- Tests: `pytest`
- Extract training data: `python extract_training_data.py`
- Pipeline: see README §9 (Quick Start)
