"""Allow `python -m lsi_vlm.audit` / `python -m lsi_vlm.stack` via package entry."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "lsi_vlm Phase 0 modules:\n"
        "  python -m lsi_vlm.audit\n"
        "  python -m lsi_vlm.stack\n"
        "Stage A/B not started from this entrypoint.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
