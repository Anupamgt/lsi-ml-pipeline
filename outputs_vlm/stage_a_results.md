# Stage A — Zero-shot VLM results

Config: `configs\vlm_stage_a.yaml`  
Backend: `ollama` / model `qwen2.5vl`  
Dry-run / mock path: **no**

## Locked FR/ML baselines

Canonical numbers from `outputs_vlm/baseline.md` (do **not** cite RF 0.8275):

| Model | Buffered LOOCV AUC (r = 1 km) |
|---|---|
| Random Forest | **0.8337** |
| Logistic Regression | **0.7707** |

## VLM ROC-AUC vs locked baselines

Primary reportable VLM row is the **no-LSI** ablation (honest baseline; with-LSI can read FR zones back from the image).

| System | Ablation | AUC (`score`) | AUC (ordinal `class`) | Notes |
|---|---|---|---|---|
| RF (locked) | — | **0.8337** | — | buffered LOOCV |
| LR (locked) | — | **0.7707** | — | buffered LOOCV |
| VLM | `no_lsi_numbers` | 0.4070 ± 0.0000 | 0.4070 ± 0.0000 | **honest baseline (report this)** |
| VLM | `with_lsi_numbers` | 0.4504 ± 0.0000 | 0.4153 ± 0.0000 |  |
| VLM | `no_lsi_image_only` | 0.4773 ± 0.0000 | 0.5000 ± 0.0000 |  |
| VLM | `with_lsi_image_only` | 0.6890 ± 0.0000 | 0.6498 ± 0.0000 |  |

## Ablation detail (required)

1. **With vs without LSI-zone overlay** — no-LSI is the honest baseline.
2. **Numbers in prompt vs image-only** — tests text leakage vs pixels.
3. **Imagery date / post-event flags** — recorded per point when known (see examples + prediction JSON).

### Per-repeat AUCs

| Ablation | Repeat | AUC score | AUC class | n_scored | n_errors | mock |
|---|---:|---:|---:|---:|---:|:---:|
| `no_lsi_numbers` | 1 | 0.4070 | 0.4070 | 44 | 0 | N |
| `no_lsi_numbers` | 2 | 0.4070 | 0.4070 | 44 | 0 | N |
| `no_lsi_numbers` | 3 | 0.4070 | 0.4070 | 44 | 0 | N |
| `with_lsi_numbers` | 1 | 0.4504 | 0.4153 | 44 | 0 | N |
| `with_lsi_numbers` | 2 | 0.4504 | 0.4153 | 44 | 0 | N |
| `with_lsi_numbers` | 3 | 0.4504 | 0.4153 | 44 | 0 | N |
| `no_lsi_image_only` | 1 | 0.4773 | 0.5000 | 44 | 0 | N |
| `no_lsi_image_only` | 2 | 0.4773 | 0.5000 | 44 | 0 | N |
| `no_lsi_image_only` | 3 | 0.4773 | 0.5000 | 44 | 0 | N |
| `with_lsi_image_only` | 1 | 0.6890 | 0.6498 | 44 | 0 | N |
| `with_lsi_image_only` | 2 | 0.6890 | 0.6498 | 44 | 0 | N |
| `with_lsi_image_only` | 3 | 0.6890 | 0.6498 | 44 | 0 | N |

## Imagery date / post-event flags

- Distinct imagery windows: `2020-11-01→2024-02-28`
- Training points with matched inventory `event_date`: **5**
- post_event=True: **5**, False: **0**, unknown: **39**
- `post_event=True` when the satellite composite window ends on/after the matched event date (scar-detection risk).

## Stability

Configured repeats are aggregated as mean ± sample stdev above. With temperature 0 and response caching, variance is often ~0.

## How to reproduce

```text
python -m lsi_vlm.eval_stage_a --config configs/vlm_stage_a.yaml --backend ollama
```

If Ollama is down and no API keys are set, add `--dry-run` (or let the harness auto-fallback to `mock`). **Mock AUCs are scaffolding only** — cite live backend numbers for the paper, always next to RF **0.8337** / LR **0.7707**.
