"""Vision-Language Model extension for LSI (complements FR/ML; Aizawl is test-only)."""

__version__ = "0.1.0"

from lsi_vlm.prompts import (  # noqa: F401
    VLMPrediction,
    build_prompt,
    parse_vlm_response,
)
from lsi_vlm.clients import (  # noqa: F401
    VLMClient,
    make_client,
    client_from_config,
)

# Locked Phase 0 benchmarks for Stage A / B comparisons (outputs_vlm/baseline.md).
LOCKED_RF_AUC = 0.8337
LOCKED_LR_AUC = 0.7707
