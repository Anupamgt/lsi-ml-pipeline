"""Allow `python -m lsi_vlm` package entry (lists Stage A commands)."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "lsi_vlm modules:\n"
        "  python -m lsi_vlm.audit\n"
        "  python -m lsi_vlm.stack\n"
        "  python -m lsi_vlm.eval_stage_a --config configs/vlm_stage_a.yaml --backend ollama\n"
        "  python -m lsi_vlm.eval_stage_a --config configs/vlm_stage_a.yaml --dry-run\n"
        "Locked baselines: RF 0.8337 / LR 0.7707 (outputs_vlm/baseline.md).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
