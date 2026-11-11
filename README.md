# Decision Models for Business Customer-Service Triage

Code, data and results for the paper *Decision Models for Business Customer-Service Triage: Accuracy, Calibration, Out-of-Scope Detection and Cost of Jev, Clef and Perplexity Decider*.

## Contents

| Path | Content |
|---|---|
| `bp01/` | Data preparation, request records, model runners, metrics, statistics and analysis |
| `bp01/runners/` | Jev and Perplexity Decisions API client, Clef, Perplexity Decider, vLLM readout and generation, exact readout, trained baselines |
| `tests/` | Unit tests (metrics, statistics, readout, API parsing) |
| `data/tasks/` | Train and test sets of BANKING77, Bitext and CLINC150 as used in the study, with `manifest.json`; `manifest_hint3.json` lists the sources, sizes and SHA-256 hashes of the HINT3 sets (SOFMattress, Curekart, Powerplay11) |
| `results/main/` | Main run: every prediction of every system and condition (`raw/<model>/<task>/<condition>.jsonl.gz`) and the result tables (`tables/`) |
| `results/round1/` | HINT3 evaluation, BF16 and exact readouts of Qwen3.8-27B (`raw/`), and the held-out threshold, out-of-scope, temperature-scaling, backbone and Bitext-overlap analyses (`tables/`) |
| `scripts/` | Environment setup, model download and the run launchers |

## Setup

Three Python 3.12 environments are used, because the models pin different library versions:

```bash
tcsh scripts/setup_server.csh
tcsh scripts/download_models.csh
```

The GPU stack (`requirements-cuda.txt`) is pinned to CUDA 12.9 builds of torch 2.13.0 and vLLM 0.31.0. The experiments ran on NVIDIA H100 NVL GPUs.

API keys are read from environment variables only:

```bash
export TYPESAFE_API_KEY=...
export PERPLEXITY_API_KEY=...
```

## Running

Main run (BANKING77, Bitext, CLINC150; all conditions):

```bash
bash scripts/launch_tmux.sh prepare
bash scripts/launch_tmux.sh full
bash scripts/launch_tmux.sh analyze
```

Additional experiments (HINT3 with all systems; Qwen3.8-27B BF16 readout on BANKING77 and the fixed 500-request subsets of Bitext and CLINC150; exact readout without top-64 truncation or pruning on the same subsets):

```bash
bash scripts/round1.sh prepare
bash scripts/round1.sh full
bash scripts/round1.sh status
```

`round1.sh` uses two GPUs (`GA`, `GB`; defaults 2 and 3). All runners are resumable. HINT3 is not redistributed here, because some of its messages contain customers' personal data such as email addresses; `python -m bp01.data --hint3` (run by `round1.sh prepare`) downloads it from the pinned commit and reproduces the files byte for byte, which the hashes in `manifest_hint3.json` confirm. `python -m bp01.data` re-downloads the source datasets and rebuilds `data/tasks/` (`--hint3` for HINT3); the test-set hashes in the manifests identify the exact splits.

## Reproducing the tables from the stored predictions

```bash
find results -name "*.jsonl.gz" -exec gunzip {} +
python -m bp01.data --hint3
python -m bp01.records
python -m bp01.records --extra
python -m bp01.analyze --run-name main
python -m bp01.round1_offline
python -m pytest tests
```

The request records are deterministic and are rebuilt from `data/tasks/` (`--extra` adds the HINT3 records and the fixed subsets used by the BF16 and exact readouts). `bp01.round1_offline` writes:

| Table | Content |
|---|---|
| `coverage_tieaware.csv` | Coverage at 2% risk with tied confidence values accepted or rejected together |
| `coverage_heldout.csv` | Threshold chosen on a calibration half, coverage and error on the other half (1,000 splits) |
| `oos_heldout.csv` | CLINC150 out-of-scope recall at 1% and 5% false alarms (held-out thresholds) |
| `temperature.csv` | ECE before and after temperature scaling; Perplexity Decider without its saved temperature |
| `bf16_vs_fp8.csv`, `bf16_deltas.csv` | Qwen3.8-27B readout, FP8 vs BF16, and the decision models against both |
| `exact_vs_approx.csv` | Exact readout vs the top-64 + pruning approximation, from the same logits |
| `hint3.csv`, `hint3_vs_readout.csv` | HINT3 results and paired comparisons with the Qwen3.8-27B readout |
| `bitext_overlap*.csv` | Nearest training neighbour of every Bitext test request and accuracy by overlap |

## Model versions

| System | Version |
|---|---|
| Cloudflare/clef | `0b331204bb13fbd2ca93a64df1956f5b55478ce5` |
| Cloudflare/clef-flash | `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` |
| perplexity-ai/pplx-decider-v1.1-27b | `6195aa55a72ae21e503c8a60198f7b390d42adb5` |
| Qwen/Qwen3.8-27B-FP8 | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Qwen/Qwen3.8-27B (BF16) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| google/gemma-4-31B-it | `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| answerdotai/ModernBERT-large | `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` |
| sentence-transformers/all-MiniLM-L6-v2 (Bitext overlap audit) | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Jev API | `jev-1.13.0` |
| Perplexity Decisions API | `pplx-decider-v1.1-27b` |

The hosted APIs can change behind an unchanged version identifier. The main run queried them on 11–12 October 2026 and the HINT3 run on 5 November 2026.

## Data licenses

BANKING77: CC BY 4.0 (PolyAI). Bitext customer-support dataset: CDLA-Sharing-1.0 (Bitext). CLINC150: CC BY 3.0 (Clinc). HINT3: Open Database License (ODbL) 1.0, contents under the Database Contents License (DbCL) 1.0 (Haptik; Arora et al., 2020, https://github.com/hellohaptik/HINT3, commit `99965c892ea9a6801a87083d3861b0b04c91a4e7`, version 2).
