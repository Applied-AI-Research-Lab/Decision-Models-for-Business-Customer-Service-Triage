from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = PROJECT_ROOT / "data"
TASKS_DIR = DATA_DIR / "tasks"
RECORDS_DIR = DATA_DIR / "records"
RESULTS_ROOT = PROJECT_ROOT / "results"

SEED = 20261010

TASKS = ["banking77", "bitext", "clinc150"]

ALL_CONDITIONS = ["canonical", "reversed", "shuffled", "no_oos", "repeat1", "repeat2"]
ROBUSTNESS_N = 500
BITEXT_TEST_N = 2000
BITEXT_TRAIN_CAP = 15000

OOS_LABEL = "oos"

HINT3_COMMIT = "99965c892ea9a6801a87083d3861b0b04c91a4e7"
HINT3_BOTS = {"sofmattress": "a mattress retailer", "curekart": "an online health and nutrition store",
              "powerplay11": "a fantasy-sports gaming app"}
HINT3_TASKS = [f"hint3_{b}" for b in HINT3_BOTS]
SUBSET_CONDITION = "subset"

JEV = {
    "base_url": os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1/systemone"),
    "model": os.environ.get("JEV_MODEL", "jev-1.13.0"),
    "api_key_env": "TYPESAFE_API_KEY",
    "usd_per_m_input": 0.042,
    "concurrency": int(os.environ.get("JEV_CONCURRENCY", "12")),
    "timeout_s": 60.0,
    "max_retries": 8,
}

PPLX_API = {
    "base_url": os.environ.get("PPLX_BASE_URL", "https://api.perplexity.ai/v1/decisions"),
    "model": os.environ.get("PPLX_MODEL", "pplx-decider-v1.1-27b"),
    "api_key_env": "PERPLEXITY_API_KEY",
    "usd_per_m_input": 0.02,
    "concurrency": int(os.environ.get("PPLX_CONCURRENCY", "8")),
    "timeout_s": 90.0,
    "max_retries": 8,
}
API_PROVIDERS = {"jev": JEV, "pplx_api": PPLX_API}

HF_MODELS = {
    "clef": "Cloudflare/clef",
    "clef_flash": "Cloudflare/clef-flash",
    "pplx": "perplexity-ai/pplx-decider-v1.1-27b",
    "qwen": "Qwen/Qwen3.8-27B-FP8",
    "qwen_bf16": "Qwen/Qwen3.8-27B",
    "gemma": "google/gemma-4-31B-it",
    "modernbert": "answerdotai/ModernBERT-large",
    "mac_readout": "Qwen/Qwen3-0.6B",
}
HF_REVISIONS = {"clef": "0b331204bb13fbd2ca93a64df1956f5b55478ce5",
                "clef_flash": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
                "pplx": "6195aa55a72ae21e503c8a60198f7b390d42adb5",
                "gemma": "842da3794eaa0b77d5f08bae87a17459d91ff475",
                "modernbert": "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
                "qwen": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
                "qwen_bf16": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}

READOUT_TOP_LOGPROBS = 64
READOUT_PRUNE = 1e-5
VLLM_MAX_MODEL_LEN = 8192
VLLM_GPU_MEM = float(os.environ.get("VLLM_GPU_MEM", "0.90"))

GPU_HOUR_USD = 2.50

TOST_MARGIN = 0.02
BOOTSTRAP_N = 10000


def results_dir(run_name: str = "main") -> Path:
    return RESULTS_ROOT / run_name


def raw_path(run_name: str, model: str, task: str, condition: str) -> Path:
    return results_dir(run_name) / "raw" / model / task / f"{condition}.jsonl"


def records_path(task: str, condition: str) -> Path:
    return RECORDS_DIR / task / f"{condition}.jsonl"


def conditions_for(task: str, model: str) -> list[str]:
    if task in HINT3_TASKS:
        return ["canonical"]
    if model == "qwen_bf16_readout":
        return ["canonical"] if task == "banking77" else [SUBSET_CONDITION]
    if model.endswith("_exact"):
        return [SUBSET_CONDITION]
    conds = ["canonical", "reversed", "shuffled"]
    if task == "clinc150":
        conds.append("no_oos")
    if model == "jev":
        conds += ["repeat1", "repeat2"]
    if model == "pplx_api":
        conds = ["canonical"]
    if model in ("modernbert", "tfidf"):
        conds = ["canonical"]
    return conds
