from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import stats as S
from .config import (API_PROVIDERS, BOOTSTRAP_N, GPU_HOUR_USD, OOS_LABEL, TASKS, TOST_MARGIN, records_path,
                     results_dir)
from .utils import get_logger, iter_jsonl, latest_ok, read_jsonl

log = get_logger("analyze")
REFERENCES = ["qwen_readout", "jev"]
API_MODELS = set(API_PROVIDERS)
DECISION_MODELS = ["jev", "clef", "clef_flash", "pplx", "pplx_api"]
HOSTED_LOCAL_PAIRS = [("pplx_api", "pplx")]


def discover(run: str) -> list[str]:
    raw = results_dir(run) / "raw"
    return sorted(p.name for p in raw.iterdir() if p.is_dir()) if raw.exists() else []


def load(run, model, task, cond):
    rp = records_path(task, cond)
    pp = results_dir(run) / "raw" / model / task / f"{cond}.jsonl"
    if not rp.exists() or not pp.exists():
        return None
    recs = read_jsonl(rp)
    preds = latest_ok(pp)
    n_err = sum(1 for r in iter_jsonl(pp) if r.get("error") and r["id"] not in preds)
    recs = [r for r in recs if r["id"] in preds]
    if not recs:
        return None
    opts = recs[0]["options"]
    P = np.array([[preds[r["id"]]["probs"].get(o, 0.0) for o in opts] for r in recs], float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    y = np.array([opts.index(r["gold"]) if r["gold"] in opts else -1 for r in recs])
    return {"recs": recs, "preds": preds, "opts": opts, "P": P, "y": y, "n_err": n_err,
            "ids": [r["id"] for r in recs]}


def cost_per_1k(model, preds: dict) -> float:
    rows = list(preds.values())
    if model in API_MODELS:
        toks = [r["input_tokens"] or 0 for r in rows]
        return float(np.mean(toks)) * API_PROVIDERS[model]["usd_per_m_input"] / 1e6 * 1000
    lat = [r["latency_s"] or 0 for r in rows]
    return float(np.mean(lat)) * GPU_HOUR_USD / 3600 * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    args = ap.parse_args()
    run = args.run_name
    out_t = results_dir(run) / "tables"
    out_f = results_dir(run) / "figures"
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    models = discover(run)
    log.info("models found: %s", models)

    cov, main_rows, oos_rows, rob_rows, eff_rows = [], [], [], [], []
    cache = {}
    for m in models:
        for t in TASKS:
            for c in ["canonical", "reversed", "shuffled", "no_oos", "repeat1", "repeat2"]:
                d = load(run, m, t, c)
                if d is None:
                    continue
                cache[(m, t, c)] = d
                n_rec = len(read_jsonl(records_path(t, c)))
                cov.append({"model": m, "task": t, "condition": c, "records": n_rec,
                            "answered": len(d["recs"]), "unresolved_errors": d["n_err"]})

    for (m, t, c), d in cache.items():
        if c != "canonical":
            continue
        P, y = d["P"], d["y"]
        row = {"model": m, "task": t, **M.summary(P, y)}
        if any("mass" in p.get("meta", {}) for p in d["preds"].values()):
            row["readout_mass_median"] = float(np.median([p["meta"].get("mass", np.nan) for p in d["preds"].values()]))
        if any("valid" in p.get("meta", {}) for p in d["preds"].values()):
            row["generative_valid_rate"] = float(np.mean([bool(p["meta"].get("valid")) for p in d["preds"].values()]))
        main_rows.append(row)
        lat = np.array([p["latency_s"] or 0 for p in d["preds"].values()])
        eff_rows.append({"model": m, "task": t, "latency_mean_s": lat.mean(), "latency_p50_s": np.median(lat),
                         "latency_p95_s": np.percentile(lat, 95),
                         "input_tokens_mean": np.mean([p["input_tokens"] or 0 for p in d["preds"].values()]),
                         "usd_per_1k": cost_per_1k(m, d["preds"])})
        if t == "clinc150":
            oi = d["opts"].index(OOS_LABEL)
            is_oos = y == oi
            pred = P.argmax(1)
            oos_rows.append({"model": m, "condition": "canonical (oos option)",
                             "in_scope_acc": float(np.mean(pred[~is_oos] == y[~is_oos])),
                             "oos_recall": float(np.mean(pred[is_oos] == oi)),
                             "oos_auroc": M.auroc(P[:, oi], is_oos)})
            dn = cache.get((m, t, "no_oos"))
            if dn is not None:
                yy = dn["y"]
                oos2 = yy == -1
                pn = dn["P"].argmax(1)
                oos_rows.append({"model": m, "condition": "no_oos (threshold on max prob)",
                                 "in_scope_acc": float(np.mean(pn[~oos2] == yy[~oos2])),
                                 "oos_recall": float("nan"),
                                 "oos_auroc": M.auroc(1 - dn["P"].max(1), oos2)})

    for (m, t, c), d in cache.items():
        if c not in ("reversed", "shuffled", "repeat1", "repeat2"):
            continue
        base = cache.get((m, t, "canonical"))
        if base is None:
            continue
        ids = [i for i in d["ids"] if i in base["preds"]]
        opts = base["opts"]
        Pa = np.array([[base["preds"][i]["probs"].get(o, 0) for o in opts] for i in ids])
        Pb = np.array([[d["preds"][i]["probs"].get(o, 0) for o in opts] for i in ids])
        rob_rows.append({"model": m, "task": t, "condition": c, "n": len(ids),
                         "flip_rate": M.flip_rate(Pa.argmax(1), Pb.argmax(1)),
                         "mean_tv": M.total_variation(Pa, Pb)})

    hl_rows = []
    for hosted, local in HOSTED_LOCAL_PAIRS:
        for t in TASKS:
            a, b = cache.get((hosted, t, "canonical")), cache.get((local, t, "canonical"))
            if a is None or b is None:
                continue
            ids = sorted(set(a["ids"]) & set(b["ids"]))
            opts = b["opts"]
            Pa = np.array([[a["preds"][i]["probs"].get(o, 0) for o in opts] for i in ids])
            Pb = np.array([[b["preds"][i]["probs"].get(o, 0) for o in opts] for i in ids])
            hl_rows.append({"hosted": hosted, "local": local, "task": t, "n": len(ids),
                            "decision_agreement": 1 - M.flip_rate(Pa.argmax(1), Pb.argmax(1)),
                            "mean_tv": M.total_variation(Pa, Pb),
                            "max_abs_prob_diff": float(np.abs(Pa - Pb).max()) if len(ids) else float("nan")})

    comp = []
    for ref in REFERENCES:
        for t in TASKS:
            r = cache.get((ref, t, "canonical"))
            if r is None:
                continue
            for m in models:
                if m == ref or (m, t, "canonical") not in cache:
                    continue
                d = cache[(m, t, "canonical")]
                common = sorted(set(d["ids"]) & set(r["ids"]))
                pos_a = {i: k for k, i in enumerate(d["ids"])}
                pos_b = {i: k for k, i in enumerate(r["ids"])}
                ia = [pos_a[i] for i in common]
                ib = [pos_b[i] for i in common]
                Pa, ya, Pb, yb = d["P"][ia], d["y"][ia], r["P"][ib], r["y"][ib]
                ca, cb = Pa.argmax(1) == ya, Pb.argmax(1) == yb
                diff, lo, hi = S.bootstrap_diff_ci(ca, cb, n_boot=args.n_boot)
                _, lo90, hi90 = S.bootstrap_diff_ci(ca, cb, n_boot=args.n_boot, level=0.90)
                ece_d = S.paired_bootstrap(lambda ix: M.ece_equal_mass(Pa[ix], ya[ix]) - M.ece_equal_mass(Pb[ix], yb[ix]),
                                           len(common), n_boot=args.n_boot)
                comp.append({"reference": ref, "model": m, "task": t, "n": len(common),
                             "acc_diff": diff, "acc_lo": lo, "acc_hi": hi,
                             "p_mcnemar": S.mcnemar_exact(ca, cb),
                             "equivalent_2pp": S.tost_equivalent(lo90, hi90, TOST_MARGIN),
                             "ece_diff": ece_d[0], "ece_lo": ece_d[1], "ece_hi": ece_d[2]})
    comp_df = pd.DataFrame(comp)
    if len(comp_df):
        comp_df["p_holm"] = np.nan
        for ref, g in comp_df.groupby("reference"):
            comp_df.loc[g.index, "p_holm"] = S.holm(g["p_mcnemar"].tolist())

    for name, rows in [("coverage", cov), ("main", main_rows), ("oos", oos_rows),
                       ("robustness", rob_rows), ("efficiency", eff_rows), ("hosted_vs_local", hl_rows)]:
        pd.DataFrame(rows).to_csv(out_t / f"{name}.csv", index=False)
    comp_df.to_csv(out_t / "comparisons.csv", index=False)
    log.info("tables written to %s", out_t)
    make_figures(cache, pd.DataFrame(main_rows), pd.DataFrame(eff_rows), out_f)


def make_figures(cache, main_df, eff_df, out_f: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if main_df.empty:
        return

    fig, axes = plt.subplots(1, len(TASKS), figsize=(5 * len(TASKS), 4), squeeze=False)
    for ax, t in zip(axes[0], TASKS):
        g = main_df[main_df.task == t]
        ax.scatter(g.ece, g.accuracy)
        for _, r in g.iterrows():
            ax.annotate(r.model, (r.ece, r.accuracy), fontsize=7)
        ax.set_title(t); ax.set_xlabel("ECE (lower is better)"); ax.set_ylabel("Accuracy")
    fig.tight_layout(); fig.savefig(out_f / "accuracy_vs_ece.png", dpi=200); plt.close(fig)

    for t in TASKS:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        for (m, tt, c), d in cache.items():
            if tt != t or c != "canonical":
                continue
            conf = d["P"].max(1); corr = (d["P"].argmax(1) == d["y"]).astype(float)
            bins = np.array_split(np.argsort(conf), 15)
            ax.plot([conf[b].mean() for b in bins if len(b)], [corr[b].mean() for b in bins if len(b)],
                    marker="o", ms=3, label=m)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_title(f"Reliability: {t}")
        ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(out_f / f"reliability_{t}.png", dpi=200); plt.close(fig)

    from sklearn.metrics import roc_curve
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, cond, title in [(axes[0], "canonical", "with 'oos' option: P(oos)"),
                            (axes[1], "no_oos", "without it: 1 - max probability")]:
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        for (m, t, c), d in sorted(cache.items()):
            if t != "clinc150" or c != cond:
                continue
            if cond == "canonical":
                oi = d["opts"].index(OOS_LABEL)
                pos, score = d["y"] == oi, d["P"][:, oi]
            else:
                pos, score = d["y"] == -1, 1 - d["P"].max(1)
            if pos.min() == pos.max():
                continue
            fpr, tpr, _ = roc_curve(pos.astype(int), score)
            ax.plot(fpr, tpr, lw=1, label=f"{m} ({M.auroc(score, pos):.3f})")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("OOS recall"); ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6, loc="lower right")
    fig.suptitle("Out-of-scope detection, CLINC150"); fig.tight_layout()
    fig.savefig(out_f / "oos_roc.png", dpi=200); plt.close(fig)

    if not eff_df.empty:
        merged = eff_df.merge(main_df[["model", "task", "accuracy"]], on=["model", "task"])
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, g in merged.groupby("model"):
            ax.scatter(g.usd_per_1k.clip(lower=1e-6), g.accuracy, label=m)
        ax.set_xscale("log"); ax.set_xlabel("$ per 1,000 decisions (log)"); ax.set_ylabel("Accuracy")
        ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(out_f / "cost_vs_accuracy.png", dpi=200); plt.close(fig)
    log.info("figures written to %s", out_f)

if __name__ == "__main__":
    main()
