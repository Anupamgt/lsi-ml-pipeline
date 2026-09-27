"""Thin entrypoint for Stage A VLM eval (wraps ``lsi_vlm.eval_stage_a``)."""

from __future__ import annotations

from lsi_vlm.eval_stage_a import main

if __name__ == "__main__":
    raise SystemExit(main())
