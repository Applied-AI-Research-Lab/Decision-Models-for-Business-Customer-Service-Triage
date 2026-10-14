from __future__ import annotations

import json
import random
from collections import Counter

from ..config import HF_MODELS, SEED, TASKS_DIR
from ..utils import Timer, append_jsonl, get_logger, read_jsonl
from .common import base_parser, jobs, pred_row

log = get_logger("baselines")


def _train(task):
    return read_jsonl(TASKS_DIR / task / "train.jsonl"), json.loads((TASKS_DIR / task / "labels.json").read_text())


def run_majority(task, recs, out):
    train, labels = _train(task)
    freq = Counter(r["label"] for r in train)
    total = sum(freq.values())
    rows = []
    for r in recs:
        probs = {o: freq.get(o, 0) / total for o in r["options"]}
        s = sum(probs.values()) or 1.0
        rows.append(pred_row(r, "majority", {k: v / s for k, v in probs.items()}, 0.0))
    append_jsonl(out, rows)


def run_mock(task, recs, out):
    rng = random.Random(SEED)
    rows = []
    for r in recs:
        w = [rng.random() for _ in r["options"]]
        s = sum(w)
        rows.append(pred_row(r, "mock", {o: x / s for o, x in zip(r["options"], w)}, 0.001))
    append_jsonl(out, rows)


def run_tfidf(task, recs, out):
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train, _ = _train(task)
    xw = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    xc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    texts = [r["text"] for r in train]
    X = hstack([xw.fit_transform(texts), xc.fit_transform(texts)]).tocsr()
    clf = LogisticRegression(max_iter=3000, C=10.0, random_state=SEED)
    clf.fit(X, [r["label"] for r in train])
    with Timer() as t:
        Xt = hstack([xw.transform([r["text"] for r in recs]), xc.transform([r["text"] for r in recs])]).tocsr()
        P = clf.predict_proba(Xt)
    per = t.seconds / len(recs)
    rows = []
    for r, p in zip(recs, P):
        d = dict(zip(clf.classes_, p))
        probs = {o: float(d.get(o, 0.0)) for o in r["options"]}
        rows.append(pred_row(r, "tfidf", probs, per))
    append_jsonl(out, rows)


def run_modernbert(task, recs, out, epochs=3, lr=5e-5, bs=32, max_len=128):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(SEED)
    train, labels = _train(task)
    lab2id = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(HF_MODELS["modernbert"])
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODELS["modernbert"], num_labels=len(labels)).cuda()
    data = [(r["text"], lab2id[r["label"]]) for r in train]

    def collate(b):
        enc = tok([x for x, _ in b], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in b])
        return enc

    dl = DataLoader(data, batch_size=bs, shuffle=True, collate_fn=collate,
                    generator=torch.Generator().manual_seed(SEED))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * len(dl) * epochs), len(dl) * epochs)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for batch in dl:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item()
        log.info("%s epoch %d loss %.4f", task, ep, tot / len(dl))
    model.eval()
    rows = []
    for i in range(0, len(recs), 128):
        chunk = recs[i:i + 128]
        enc = tok([r["text"] for r in chunk], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt").to("cuda")
        with Timer() as t, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            P = model(**enc).logits.float().softmax(-1).cpu().tolist()
        for r, p in zip(chunk, P):
            probs = {o: p[lab2id[o]] for o in r["options"]}
            rows.append(pred_row(r, "modernbert", probs, t.seconds / len(chunk)))
    append_jsonl(out, rows)

RUNNERS = {"tfidf": run_tfidf, "modernbert": run_modernbert, "majority": run_majority, "mock": run_mock}


def main():
    ap = base_parser("Supervised/trivial baselines")
    ap.add_argument("--which", choices=list(RUNNERS), required=True)
    args = ap.parse_args()
    if args.which != "mock":
        args.conditions = "canonical"
    for task, cond, recs, out in jobs(args, args.which):
        RUNNERS[args.which](task, recs, out)
        log.info("done %s", out)

if __name__ == "__main__":
    main()
