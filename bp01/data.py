from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from .config import BITEXT_TEST_N, BITEXT_TRAIN_CAP, OOS_LABEL, SEED, TASKS_DIR
from .utils import get_logger, write_jsonl

log = get_logger("data")

BANKING_URLS = {
    "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}
BITEXT_REPO = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
BITEXT_FILE = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
CLINC_REPO = "clinc/clinc_oos"


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(task: str, train: list[dict], test: list[dict], labels: list[str], source: str, limit: int | None):
    d = TASKS_DIR / task
    if limit:
        test = test[:limit]
        train = train[: max(limit * 5, 200)]
    write_jsonl(d / "train.jsonl", train)
    write_jsonl(d / "test.jsonl", test)
    (d / "labels.json").write_text(json.dumps(labels, indent=1))
    log.info("%s: train=%d test=%d labels=%d", task, len(train), len(test), len(labels))
    return {
        "task": task, "source": source, "n_train": len(train), "n_test": len(test),
        "n_labels": len(labels), "sha256_test": _sha(d / "test.jsonl"), "limited": bool(limit),
    }


def prepare_banking77(limit=None):
    import pandas as pd
    try:
        tr = pd.read_csv(BANKING_URLS["train"])
        te = pd.read_csv(BANKING_URLS["test"])
        src = "PolyAI-LDN/task-specific-datasets (GitHub CSV), CC BY 4.0"
    except Exception as e:
        log.warning("GitHub CSV failed (%s); using HF parquet branch", e)
        from datasets import load_dataset
        ds = load_dataset("PolyAI/banking77", revision="refs/convert/parquet")
        names = ds["train"].features["label"].names
        tr = pd.DataFrame({"text": ds["train"]["text"], "category": [names[i] for i in ds["train"]["label"]]})
        te = pd.DataFrame({"text": ds["test"]["text"], "category": [names[i] for i in ds["test"]["label"]]})
        src = "PolyAI/banking77 (HF parquet), CC BY 4.0"
    labels = sorted(tr["category"].unique().tolist())
    train = [{"id": f"banking77-train-{i:05d}", "text": t, "label": c} for i, (t, c) in enumerate(zip(tr.text, tr.category))]
    test = [{"id": f"banking77-test-{i:05d}", "text": t, "label": c} for i, (t, c) in enumerate(zip(te.text, te.category))]
    return _save("banking77", train, test, labels, src, limit)


def prepare_bitext(limit=None):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from sklearn.model_selection import train_test_split

    path = hf_hub_download(BITEXT_REPO, BITEXT_FILE, repo_type="dataset")
    df = pd.read_csv(path)[["instruction", "intent"]].dropna()
    df = df.drop_duplicates(subset=["instruction"]).reset_index(drop=True)
    tr, te = train_test_split(df, test_size=0.2, stratify=df["intent"], random_state=SEED)

    train_norm = set(tr["instruction"].map(_norm))
    te = te[~te["instruction"].map(_norm).isin(train_norm)]
    te, _ = train_test_split(te, train_size=min(BITEXT_TEST_N, len(te) - 27), stratify=te["intent"], random_state=SEED)
    if len(tr) > BITEXT_TRAIN_CAP:
        tr, _ = train_test_split(tr, train_size=BITEXT_TRAIN_CAP, stratify=tr["intent"], random_state=SEED)
    labels = sorted(df["intent"].unique().tolist())
    train = [{"id": f"bitext-train-{i:05d}", "text": t, "label": c} for i, (t, c) in enumerate(zip(tr.instruction, tr.intent))]
    test = [{"id": f"bitext-test-{i:05d}", "text": t, "label": c} for i, (t, c) in enumerate(zip(te.instruction, te.intent))]
    return _save("bitext", train, test, labels, f"{BITEXT_REPO} ({BITEXT_FILE}), CDLA-Sharing-1.0", limit)


def prepare_clinc(limit=None):
    from datasets import load_dataset
    ds = load_dataset(CLINC_REPO, "plus")
    names = ds["train"].features["intent"].names

    def rows(split):
        return [{"id": f"clinc150-{split}-{i:05d}", "text": t, "label": names[y]}
                for i, (t, y) in enumerate(zip(ds[split]["text"], ds[split]["intent"]))]

    train, test = rows("train"), rows("test")
    in_scope = sorted(n for n in names if n != OOS_LABEL)
    labels = in_scope + [OOS_LABEL]
    if limit:
        oos = [r for r in test if r["label"] == OOS_LABEL][: max(2, limit // 5)]
        test = [r for r in test if r["label"] != OOS_LABEL][: limit - len(oos)] + oos
    return _save("clinc150", train, test, labels, f"{CLINC_REPO} (plus), CC BY 3.0", limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="tiny sample for a quick test")
    args = ap.parse_args()
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = [prepare_banking77(args.limit), prepare_bitext(args.limit), prepare_clinc(args.limit)]
    (TASKS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("manifest written: %s", TASKS_DIR / "manifest.json")

if __name__ == "__main__":
    main()
