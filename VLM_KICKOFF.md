# VLM Kickoff Brief — Vision-Language Landslide Susceptibility

> **For the coding agent starting a new chat in this repo.** Read this whole file before touching code.
> Work phase by phase and stop at each **checkpoint** for the user to review. Do not skip ahead to Stage B.

---

## 1. What we are building and why

The repo already has a validated **Frequency Ratio (FR) + ML** landslide susceptibility pipeline for **Aizawl, Mizoram**:
17 conditioning factors, 22 mapped landslides + 22 pseudo-absences, buffered spatial LOOCV with a 1 km exclusion radius
(RF AUC 0.8275, LR AUC 0.7707).

The next step is a **vision-language model (VLM)** layer, built in two stages:

| | Stage A: zero-shot baseline | Stage B: main method |
|---|---|---|
| Input | Rendered colour tiles (hillshade, slope, LSI zones) + satellite RGB + text prompt with legend, numbers, context | All factor layers as a **continuous-valued multi-band stack** |
| Model | Off-the-shelf VLM, no training | Pretrained Earth-observation encoder → LLaVA-style connector → small LLM |
| Training | None | Fine-tune on NASA Global Landslide Catalog (+ FR/ML maps as auxiliary labels) |
| Output | Susceptibility class + short reason per tile | Susceptibility map for any region + plain-language explanation |
| Weakness | Model sees colours, not values | Needs global data, GPU, weeks of work |

**Two framing rules. Every design choice must respect them:**

1. **Aizawl is the test set, never the training set.** 22 landslides cannot fine-tune a VLM. Aizawl is the benchmark where the answer is already known (FR/ML results). In Stage B, **exclude all training samples within Aizawl's extent + a 5 km buffer**.
2. **The VLM complements FR/ML and does not replace it.** FR/ML gives checkable numbers. The VLM adds transfer to new regions and explanations. Never remove or change the behaviour of the existing pipeline.

---

## 2. Repo facts (verified, don't re-derive)

- Branch at hand-off: `feature/nhess-upgrades-portability`. Package `lsi_pipeline/`, entry `run_pipeline.py`, study config `configs/aizawl.yaml` (CRS **EPSG:32646**).
- Feature registry: `lsi_pipeline/feature_registry.py` maps 17 feature columns → class GeoTIFFs in `final_maps/sonker_17/`.
- Training table: `data/landslide_training_data.csv`, which has 44 rows (22 × `target=1`, 22 × `target=0`), columns `x, y, <17 *_fr>, target`.
- NASA Global Landslide Catalog is already in the repo: `final_maps/Global_Landslide_Catalog_Export_rows.csv` (~8.5 MB). **Inspect its columns before using it.** Don't assume a schema.
- `lsi_pipeline/pu_bagging.py` (buffer-control sampling + PU bagging) is in the tree but unvalidated. See "Repo state" below.
- Other rasters in `final_maps/`: `HILLSHADE_2.tif`, `Slope_FINAL.tif`, `DEM_UTF_FINAL.tif`, `LSI_Final_Zones.tif`, `LSI_Master.tif`, four Copernicus 30 m DEM tiles.
- `.gitignore` ignores `*.tif`, `*.gpkg`, etc. **outside** `final_maps/` and `data/`. New outputs go in `outputs_vlm/`, which is already covered by the `outputs*/` ignore rule.

### Repo state at hand-off (read carefully)

- **Commits are still being pushed** from this branch. Run `git status` and `git log origin/feature/nhess-upgrades-portability..HEAD` first. If anything is unpushed or uncommitted, **stop and tell the user**. Don't branch off a dirty tree.
- **No `outputs_*/` folder exists.** The models, ROC curves and susceptibility maps from the earlier demo run were never committed (`outputs*/` is gitignored). The quoted RF 0.8275 / LR 0.7707 are **not reproducible from repo artifacts yet**. Phase 0 must regenerate them.
- **The NHESS-style upgrades are not finished.** `pu_bagging.py` (Positive-Unlabeled bagging), `feature_selection.py` (Genetic Algorithm selection) and `train_xgboost()` in `model_training.py` exist in the working tree but are untested and unvalidated, and the current evaluation doesn't use them. **Don't depend on them** for anything the VLM work needs. If Stage B needs PU sampling, write tests for `pu_bagging.py` first and confirm with the user.
- **The pipeline is still Aizawl-only.** The FR layers are local reclassifications, so the pipeline can't move to another study area without rebuilding them. `configs/aizawl.yaml` + `study_area.py` are the start of portability work, not a finished feature. That is also why Stage B has to rebuild factors from **global** sources rather than reuse the Aizawl FR layers.

### Known data issues. Handle these in Phase 0 before any modelling.

1. **Most factor rasters are classes, not values.** Only `S17_01_Rainfall_CHIRPS_mm.tif` and `S17_10_Gravity_Bouguer_mGal.tif` are continuous. The other 15 are reclassified integers (`*_classes.tif`, `*_quantile5.tif`). Stage B's premise of "raw numbers, not colours" needs the **continuous source layers rebuilt** (slope °, elevation m, distance-to-X m, TWI, SPI, STI, TRI, NDVI, …). Categorical factors (geology, soil, geomorphology, LULC) stay categorical and should be one-hot encoded or given embeddings.
2. **The study bbox in `configs/aizawl.yaml` looks wrong.** It says x 520 000–545 000, y 2 580 000–2 610 000. The landslide points sit at x ≈ 468 800–505 200, y ≈ 2 583 000–2 637 000, which matches Aizawl (~92.7° E) in UTM 46N. **Confirm with the user before changing it**, and don't silently "fix" anything the FR/ML results depend on.
3. **Several FR features don't vary across the 44 points:** `elevation_fr`, `tri_fr`, `geology_fr`, `distance_roads_fr`. Some FR values are extreme: `lulc_fr` up to ~40.8, `ndvi_fr` up to ~340.7, likely from classes with tiny area. Report this in the Phase 0 audit. It affects how far FR maps can be trusted as auxiliary labels.
4. The pseudo-absences extend ~70 km north of the positives (y up to ~2 693 000). Note this in the audit, because it makes the classification easier than it looks.

---

## 3. Ground rules

- **New branch** `feature/vlm`, created from `feature/nhess-upgrades-portability` **only after** the user confirms the push is done and `git status` is clean. Commit per phase. Never force-push. Never push without asking.
- **New package** `lsi_vlm/` alongside `lsi_pipeline/`. Import from `lsi_pipeline`, but never modify it (except a confirmed bbox fix).
- **Optional dependencies**: add a `[project.optional-dependencies] vlm = [...]` group in `pyproject.toml`. Keep the base install unchanged.
- **Secrets** come only from env vars (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`). Never write keys to files. Add `.env` to `.gitignore`.
- **Cache every VLM call** to `outputs_vlm/cache/*.jsonl`, keyed by (tile hash, prompt hash, model). Re-runs must not re-bill.
- **Windows machine.** Use `pathlib`, no shell-specific scripts. Python ≥ 3.11.
- Config lives in `configs/vlm_stage_a.yaml` / `configs/vlm_stage_b.yaml`, not hard-coded.
- Add tests in `tests/` for tiling, prompt building, response parsing, and scoring.

---

## 4. Phase 0: baseline, audit and continuous stack (checkpoint)

0. **Regenerate the FR/ML baseline.** Follow README §9: `pip install -e .[dev]`, `python extract_training_data.py`, then the full buffered-LOOCV run into `outputs_real/`, with RF and LR only (the XGBoost/PU/GA paths stay off). Record the AUCs in `outputs_vlm/baseline.md`. If they differ from 0.8275 / 0.7707, **report the difference and stop**. Every later comparison uses these regenerated numbers.
1. `lsi_vlm/audit.py`: for every raster in `final_maps/` and `final_maps/sonker_17/`, print CRS, resolution, extent, dtype, nodata, unique-value count, and whether the raster is categorical or continuous. Write `outputs_vlm/audit.md`, including issues 1–4 above with numbers.
2. List which continuous source layers exist locally and which must be rebuilt. For each missing one, propose a source:

   | Factor | Continuous version | Rebuild from |
   |---|---|---|
   | Slope, elevation, TRI, TWI, SPI, STI | degrees / m / index | Copernicus DEM tiles already in repo (`richdem` or `whitebox`) |
   | Distance to drainage | m | Drainage network derived from the DEM |
   | Distance to faults / roads | m | Fault and road vector sources (ask user; OSM for roads) |
   | Rainfall | mm | `S17_01_Rainfall_CHIRPS_mm.tif` (exists) |
   | Gravity anomaly | mGal | `S17_10_Gravity_Bouguer_mGal.tif` (exists) |
   | Earthquake | event density / PGA | USGS catalog (ask user what was used originally) |
   | NDVI | −1…1 | Sentinel-2 composite |
   | Geology, soil, geomorphology, LULC | categorical | Keep class rasters; one-hot |

3. `lsi_vlm/stack.py`: build `outputs_vlm/aizawl_stack.tif`, a co-registered multi-band GeoTIFF (EPSG:32646, 30 m), with band names in the tags and per-band normalisation stats saved to JSON.

**Checkpoint 0:** show the audit and the stack band list. Wait for approval.

---

## 5. Stage A: zero-shot VLM baseline (checkpoint)

### A1. Tiles (`lsi_vlm/tiles.py`)
- For each of the 44 points in `data/landslide_training_data.csv`, cut a window centred on the point. Default **64 × 64 px at 30 m ≈ 1.9 km**, configurable.
- Render two PNGs per point:
  - **Composite**: hillshade (greyscale base) + slope (colour ramp, alpha) + optional LSI-zone overlay, with a fixed, documented colour legend.
  - **Satellite RGB**: Sentinel-2 L2A true colour, same window. Source it via Microsoft Planetary Computer STAC (`pystac-client` + `planetary-computer`, no account needed). Build a **dry-season median composite (Nov–Feb)**, because monsoon cloud cover over Mizoram is severe.
- Also build a coarse tile grid over the whole study area for a map-style run later. Don't run the grid until the 44-point results are reviewed.

### A2. Prompt (`lsi_vlm/prompts.py`)
The prompt includes: the colour legend and scale bar in metres, the numeric values at the tile centre from the continuous stack (elevation range, mean slope, rainfall mm, distance to drainage/faults/roads, geology/LULC class names), and short regional context (Mizoram fold belt, monsoon-triggered shallow slides). The answer must be **JSON only**:
```json
{"class": "very_low|low|moderate|high|very_high", "score": 0-100, "key_factors": ["..."], "reason": "<=60 words"}
```

### A3. Models (`lsi_vlm/clients.py`)
A single `VLMClient` interface with backends: Anthropic (Claude), Google (Gemini), OpenAI, and **local Ollama** (e.g. `qwen2.5vl`), so one free option always exists. Temperature 0. Retry and parse failures are logged, never silently dropped.

### A4. Evaluation (`lsi_vlm/eval_stage_a.py`)
- ROC-AUC of `score` (and of the ordinal `class`) against `target` over the 44 points. Put it in a table next to RF 0.8275 and LR 0.7707.
- **Required ablations**, because each one controls a leak:
  - **With vs. without the LSI-zone overlay.** With LSI zones in the image, the VLM can just read the FR answer back. The no-LSI run is the honest baseline and the one to report.
  - **Numbers in prompt vs. image only.** This shows whether the model uses the pixels or the text.
  - **Imagery date.** If the satellite image post-dates a landslide, the scar may be visible, which turns susceptibility into detection. Record the imagery date range and, where event dates are known, flag post-event tiles.
- Stability: 3 repeat runs per backend; report variance.
- Save 5 example tiles + responses to `outputs_vlm/stage_a_examples.md` for the paper.

**Checkpoint A:** results table, ablations, examples. Wait for approval before Stage B.

---

## 6. Stage B: fine-tuned geospatial VLM (design first, then build)

**Start with a design doc** (`docs/vlm_stage_b_design.md`) covering the decisions below. Get approval before writing training code.

### B1. Encoder decision
Candidates: **Prithvi-EO-2.0** (IBM/NASA; pretrained on 6 HLS spectral bands), **Clay** (Clay Foundation; wavelength-conditioned dynamic band embedding), **SatMAE** (Stanford; multispectral band groups).

Key issue: all three were pretrained on **spectral reflectance**, not on terrain/hydrology/geology factors. Feeding 17 non-spectral bands means re-initialising the patch embedding, which throws away much of the pretraining benefit. **Recommended design to evaluate: dual-branch.** The pretrained EO encoder takes the Sentinel-2/HLS bands it was trained on, and a small CNN/ViT branch takes the continuous factor stack. The two outputs are fused before the connector. Compare against a single-branch "re-initialised patch embed" baseline.

### B2. Output head vs. language
Generating a raster with an LLM is awkward and hard to validate. **Recommended:**
- A **susceptibility head** (per-tile or per-pixel logistic output) on the fused features produces the map. This is what gets AUC-evaluated.
- A **LLaVA-style MLP projector + small instruction LLM** (e.g. Qwen2.5-1.5B/3B-Instruct, LoRA) produces the explanation, conditioned on the features **and** the head's score plus the factor values. That keeps the text consistent with the number.
- Reference architecture: GeoChat (LLaVA-1.5 based, RGB only). Borrow its training recipe, not its input format.

### B3. Training data
- NASA GLC: filter to the Indian subcontinent plus analogous terrains (Himalaya, Hindu Kush, Myanmar/NE India, Western Ghats, Sri Lanka highlands) and to good `location_accuracy` (inspect the actual values first).
- Known GLC biases: news-reporting bias toward roads and towns, mostly rainfall-triggered events, point-location error. Document them and stratify evaluation by trigger.
- Negatives: buffer-control sampling (min distance from positives), either by reusing `pu_bagging.apply_buffer_control_sampling` **after** writing tests for it, or by a small, tested reimplementation in `lsi_vlm/`.
- Optional second inventory: the GSI Bhukosh national landslide inventory, if the user supplies it.
- FR/ML maps as **auxiliary** soft labels only where they exist. Down-weight them (see data issue 3).
- **Hard exclusion:** nothing inside Aizawl's extent + 5 km.
- The factor stack must be rebuilt **globally** from consistent sources (Copernicus DEM, CHIRPS, USGS, SoilGrids, GLiM geology, ESA WorldCover, Sentinel-2 NDVI, OSM). The Aizawl-specific sources (Soil_India, local geology) will not exist elsewhere. **This is the largest task in Stage B.**

### B4. Evaluation
- Primary: AUC on Aizawl's 44 points, compared directly with FR/RF/LR and Stage A.
- Secondary: spatial block hold-out regions (e.g. leave out one Himalayan block).
- Explanation quality: do the `key_factors` agree with FR weights and RF importance? Report it as a rank correlation.

### B5. Compute
LoRA on a 1.5–3B LLM plus a ~300M encoder needs a GPU with **≥ 24 GB** for comfortable batch sizes (16 GB is possible with 4-bit quantisation and small batches). The target platform comes from the user (see prep list). Write training scripts that run on Kaggle/Colab and on a Linux HPC node, and checkpoint often.

---

## 7. Deliverables checklist

- [ ] `feature/vlm` branch, `lsi_vlm/` package, `vlm` optional deps, tests passing
- [ ] `outputs_vlm/baseline.md` (regenerated RF/LR AUCs)
- [ ] `outputs_vlm/audit.md` + `aizawl_stack.tif`
- [ ] Stage A: results table, ablations, examples, cached responses
- [ ] `docs/vlm_stage_b_design.md` approved
- [ ] Stage B: global stack builder, training script, Aizawl benchmark results
- [ ] README section "Phase 3: Vision-Language Extension" with commands to reproduce

## 8. Questions to ask the user at the start
1. Which VLM backends are available (API keys set? Ollama installed with a vision model pulled?)
2. Where are the original **continuous** layers from before reclassification, and what sources were used for faults, roads, and earthquakes?
3. Is the `aizawl.yaml` bbox wrong, or intentional?
4. GPU platform for Stage B?
5. Has the push of `feature/nhess-upgrades-portability` finished, and should `feature/vlm` branch from it or from `main`?
