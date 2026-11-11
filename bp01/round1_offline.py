from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from . import metrics as M
from . import stats as S
from .analyze import load
from .config import HINT3_TASKS, OOS_LABEL, TASKS, TASKS_DIR, results_dir
from .utils import get_logger, read_jsonl

log = get_logger("round1")
MAIN_SYSTEMS = ["jev", "clef", "clef_flash", "pplx", "pplx_api", "qwen_readout", "qwen_gen", "gemma_readout",
                "gemma_gen", "modernbert", "tfidf"]
PPLX_SAVED_T = 1.0087417621345625
RISK = 0.02


def tie_aware_threshold(conf, correct, max_risk=RISK):
    lev, inv = np.unique(-conf, return_inverse=True)
    n_at = np.bincount(inv, minlength=len(lev))
    e_at = np.bincount(inv, weights=1.0 - correct, minlength=len(lev))
    acc_n, acc_e = np.cumsum(n_at), np.cumsum(e_at)
    ok = np.where(acc_e / acc_n <= max_risk + 1e-12)[0]
    if not len(ok):
        return np.inf, 0.0
    j = ok.max()
    return float(-lev[j]), float(acc_n[j] / len(conf))


def stratified_halves(y, n_splits, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    masks = np.zeros((n_splits, n), bool)
    groups = [np.where(y == c)[0] for c in np.unique(y)]
    for s in range(n_splits):
        flip = rng.integers(0, 2)
        for g in groups:
            perm = rng.permutation(g)
            k = (len(g) + flip) // 2
            flip = 1 - flip if len(g) % 2 else flip
            masks[s, perm[:k]] = True
    return masks


def heldout_coverage(conf, correct, masks):
    cov, err = [], []
    for m in masks:
        t, _ = tie_aware_threshold(conf[m], correct[m])
        acc = conf[~m] >= t
        cov.append(acc.mean())
        err.append((1 - correct[~m][acc]).mean() if acc.any() else np.nan)
    return np.array(cov), np.array(err)


def pct(a, q):
    a = a[~np.isnan(a)]
    return float(np.percentile(a, q)) if len(a) else np.nan


def fa_threshold(score_in, alpha):
    lev = np.unique(score_in)[::-1]
    s = np.sort(score_in)
    frac = 1 - np.searchsorted(s, lev, side="left") / len(s)
    ok = np.where(frac <= alpha + 1e-12)[0]
    return float(lev[ok.max()]) if len(ok) else np.inf


def fit_temperature(logp, y):
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    def nll(log_t):
        z = logp / np.exp(log_t)
        return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(y)), y]))
    r = minimize_scalar(nll, bounds=(-3, 3), method="bounded", options={"xatol": 1e-4})
    return float(np.exp(r.x))


def apply_temperature(logp, t):
    from scipy.special import softmax
    return softmax(logp / t, axis=1)


def main_task_analyses(n_splits, ts_splits, out):
    cov_rows, ho_rows, ts_rows, oos_rows = [], [], [], []
    for ti, t in enumerate(TASKS):
        base = load("main", "qwen_readout", t, "canonical")
        y = base["y"]
        masks = stratified_halves(y, n_splits, seed=1000 + ti)
        for m in MAIN_SYSTEMS:
            d = load("main", m, t, "canonical")
            assert d["ids"] == base["ids"]
            P = d["P"]
            conf, correct = P.max(1), (P.argmax(1) == y).astype(float)
            _, cov_ta = tie_aware_threshold(conf, correct)
            cov_rows.append({"model": m, "task": t, "cov_stable_sort": M.coverage_at_risk(P, y),
                             "cov_tie_aware": cov_ta, "distinct_conf": len(np.unique(conf))})
            c, e = heldout_coverage(conf, correct, masks)
            ho_rows.append({"model": m, "task": t, "splits": n_splits,
                            "cov_mean": c.mean(), "cov_lo": pct(c, 2.5), "cov_hi": pct(c, 97.5),
                            "err_mean": np.nanmean(e), "err_lo": pct(e, 2.5), "err_hi": pct(e, 97.5),
                            "share_err_above_2pct": float(np.mean(e[~np.isnan(e)] > RISK)) if np.any(~np.isnan(e)) else np.nan,
                            "share_no_threshold": float(np.mean(np.isnan(e)))})
            logp = np.log(np.clip(P, 1e-12, 1.0))
            e_raw, e_ts, temps = [], [], []
            for msk in masks[:ts_splits]:
                tt = fit_temperature(logp[msk], y[msk])
                temps.append(tt)
                e_raw.append(M.ece_equal_mass(P[~msk], y[~msk]))
                e_ts.append(M.ece_equal_mass(apply_temperature(logp[~msk], tt), y[~msk]))
            row = {"model": m, "task": t, "ts_splits": ts_splits, "T_median": float(np.median(temps)),
                   "ece_full": M.ece_equal_mass(P, y), "ece_raw_test_mean": float(np.mean(e_raw)),
                   "ece_ts_test_mean": float(np.mean(e_ts)), "ece_ts_lo": pct(np.array(e_ts), 2.5),
                   "ece_ts_hi": pct(np.array(e_ts), 97.5)}
            if m == "pplx":
                Praw = apply_temperature(logp, 1.0 / PPLX_SAVED_T)
                row["ece_without_saved_T"] = M.ece_equal_mass(Praw, y)
                row["acc_without_saved_T"] = M.accuracy(Praw, y)
            ts_rows.append(row)
            log.info("%s %s: cov %.3f -> tie-aware %.3f; held-out %.3f; ECE %.3f -> TS %.3f", t, m,
                     cov_rows[-1]["cov_stable_sort"], cov_ta, c.mean(), row["ece_full"], row["ece_ts_test_mean"])
        if t == "clinc150":
            oos_rows += oos_heldout(masks, y, base["opts"].index(OOS_LABEL))
    pd.DataFrame(cov_rows).to_csv(out / "coverage_tieaware.csv", index=False)
    pd.DataFrame(ho_rows).to_csv(out / "coverage_heldout.csv", index=False)
    pd.DataFrame(ts_rows).to_csv(out / "temperature.csv", index=False)
    pd.DataFrame(oos_rows).to_csv(out / "oos_heldout.csv", index=False)


def oos_heldout(masks, y, oi):
    rows = []
    is_oos = y == oi
    for m in MAIN_SYSTEMS:
        for cond in ["canonical", "no_oos"]:
            d = load("main", m, "clinc150", cond)
            if d is None:
                continue
            ids_c = load("main", "qwen_readout", "clinc150", "canonical")["ids"]
            assert d["ids"] == ids_c
            score = d["P"][:, oi] if cond == "canonical" else 1 - d["P"].max(1)
            row = {"model": m, "condition": cond, "auroc": M.auroc(score, is_oos)}
            for a in (0.01, 0.05):
                rec, fa = [], []
                for msk in masks:
                    th = fa_threshold(score[msk & ~is_oos], a)
                    flag = score[~msk] >= th
                    rec.append(flag[is_oos[~msk]].mean())
                    fa.append(flag[~is_oos[~msk]].mean())
                rec, fa = np.array(rec), np.array(fa)
                k = f"{int(a * 100)}pct"
                row.update({f"recall_{k}": rec.mean(), f"recall_{k}_lo": pct(rec, 2.5), f"recall_{k}_hi": pct(rec, 97.5),
                            f"fa_{k}": fa.mean(), f"fa_{k}_lo": pct(fa, 2.5), f"fa_{k}_hi": pct(fa, 97.5)})
            rows.append(row)
            log.info("OOS %s/%s: recall@1%% %.3f  recall@5%% %.3f", m, cond, row["recall_1pct"], row["recall_5pct"])
    return rows


def backbone_checks(n_boot, out):
    rows, deltas, ex_rows = [], [], []
    for t in TASKS:
        cond = "canonical" if t == "banking77" else "subset"
        b16 = load("round1", "qwen_bf16_readout", t, cond)
        f8 = load("main", "qwen_readout", t, "canonical")
        pos = {i: k for k, i in enumerate(f8["ids"])}
        ix = [pos[i] for i in b16["ids"]]
        y = b16["y"]
        assert np.array_equal(f8["y"][ix], y)
        Pf, Pb = f8["P"][ix], b16["P"]
        for name, P in [("fp8", Pf), ("bf16", Pb)]:
            rows.append({"task": t, "n": len(y), "precision": name, "accuracy": M.accuracy(P, y),
                         "ece": M.ece_equal_mass(P, y),
                         "cov_tie_aware": tie_aware_threshold(P.max(1), (P.argmax(1) == y).astype(float))[1],
                         "aurc": M.aurc(P, y)})
        rows.append({"task": t, "n": len(y), "precision": "agreement", "accuracy": float(np.mean(Pf.argmax(1) == Pb.argmax(1))),
                     "ece": M.total_variation(Pf, Pb)})
        cb, cf = Pb.argmax(1) == y, Pf.argmax(1) == y
        diff = S.bootstrap_diff_ci(cb, cf, n_boot=n_boot)
        ece_d = S.paired_bootstrap(lambda i: M.ece_equal_mass(Pb[i], y[i]) - M.ece_equal_mass(Pf[i], y[i]), len(y), n_boot=n_boot)
        deltas.append({"task": t, "model": "qwen_bf16_readout", "reference": "qwen_readout_fp8",
                       "acc_diff": diff[0], "acc_lo": diff[1], "acc_hi": diff[2], "p_mcnemar": S.mcnemar_exact(cb, cf),
                       "ece_diff": ece_d[0], "ece_lo": ece_d[1], "ece_hi": ece_d[2]})
        for m in ["clef", "clef_flash", "pplx"]:
            d = load("main", m, t, "canonical")
            Pm = d["P"][[{i: k for k, i in enumerate(d["ids"])}[i] for i in b16["ids"]]]
            cm = Pm.argmax(1) == y
            for ref, Pr in [("qwen_readout_fp8", Pf), ("qwen_bf16_readout", Pb)]:
                cr = Pr.argmax(1) == y
                dd = S.bootstrap_diff_ci(cm, cr, n_boot=n_boot)
                ed = S.paired_bootstrap(lambda i: M.ece_equal_mass(Pm[i], y[i]) - M.ece_equal_mass(Pr[i], y[i]), len(y),
                                        n_boot=n_boot)
                deltas.append({"task": t, "model": m, "reference": ref, "n": len(y),
                               "acc_diff": dd[0], "acc_lo": dd[1], "acc_hi": dd[2], "p_mcnemar": S.mcnemar_exact(cm, cr),
                               "ece_diff": ed[0], "ece_lo": ed[1], "ece_hi": ed[2]})
        ex = load("round1", "qwen_bf16_exact", t, "subset")
        opts = ex["opts"]
        Pe = ex["P"]
        Pa = np.array([ex["preds"][i]["meta"]["approx"] for i in ex["ids"]])
        ye = ex["y"]
        mass_ex = np.array([ex["preds"][i]["meta"]["mass_exact"] for i in ex["ids"]])
        mass_ap = np.array([ex["preds"][i]["meta"]["mass_approx"] for i in ex["ids"]])
        dmax = np.abs(Pe - Pa).max(1)
        pb = {i: k for k, i in enumerate(b16["ids"])}
        Pv = b16["P"][[pb[i] for i in ex["ids"]]] if all(i in pb for i in ex["ids"]) else None
        sc = [ex["preds"][i]["meta"]["self_check_max_abs_dlogprob"] for i in ex["ids"]
              if "self_check_max_abs_dlogprob" in ex["preds"][i]["meta"]]
        ex_rows.append({"task": t, "n": len(ye), "k": len(opts),
                        "argmax_changes": int(np.sum(Pe.argmax(1) != Pa.argmax(1))),
                        "max_abs_dp": float(dmax.max()), "p99_abs_dp": float(np.percentile(dmax, 99)),
                        "median_abs_dp": float(np.median(dmax)),
                        "acc_exact": M.accuracy(Pe, ye), "acc_approx": M.accuracy(Pa, ye),
                        "ece_exact": M.ece_equal_mass(Pe, ye), "ece_approx": M.ece_equal_mass(Pa, ye),
                        "mass_exact_min": float(mass_ex.min()), "mass_exact_p1": float(np.percentile(mass_ex, 1)),
                        "mass_exact_median": float(np.median(mass_ex)),
                        "mass_approx_over_exact_min": float((mass_ap / mass_ex).min()),
                        "vllm_vs_hf_agreement": float(np.mean(Pv.argmax(1) == Pe.argmax(1))) if Pv is not None else np.nan,
                        "vllm_vs_hf_mean_tv": M.total_variation(Pv, Pe) if Pv is not None else np.nan,
                        "self_check_max": float(max(sc)) if sc else np.nan})
        log.info("backbone %s: %s", t, ex_rows[-1])
    pd.DataFrame(rows).to_csv(out / "bf16_vs_fp8.csv", index=False)
    pd.DataFrame(deltas).to_csv(out / "bf16_deltas.csv", index=False)
    pd.DataFrame(ex_rows).to_csv(out / "exact_vs_approx.csv", index=False)


def hint3(n_boot, out):
    rows, comp = [], []
    systems = MAIN_SYSTEMS + ["majority"]
    for t in HINT3_TASKS:
        ref = load("round1", "qwen_readout", t, "canonical")
        y, oi = ref["y"], ref["opts"].index(OOS_LABEL)
        for m in systems:
            d = load("round1", m, t, "canonical")
            assert d["ids"] == ref["ids"]
            P = d["P"]
            pred = P.argmax(1)
            rows.append({"task": t, "model": m, "n": len(y), "k": len(ref["opts"]), "oos_share": float(np.mean(y == oi)),
                         "accuracy": M.accuracy(P, y), "macro_f1": M.macro_f1(P, y),
                         "ece": M.ece_equal_mass(P, y),
                         "cov_tie_aware": tie_aware_threshold(P.max(1), (pred == y).astype(float))[1],
                         "aurc": M.aurc(P, y),
                         "in_scope_acc": float(np.mean(pred[y != oi] == y[y != oi])),
                         "oos_recall": float(np.mean(pred[y == oi] == oi)),
                         "false_alarm": float(np.mean(pred[y != oi] == oi)),
                         "oos_auroc": M.auroc(P[:, oi], y == oi)})
            if m != "qwen_readout":
                cm, cr = pred == y, ref["P"].argmax(1) == y
                dd = S.bootstrap_diff_ci(cm, cr, n_boot=n_boot)
                comp.append({"task": t, "model": m, "acc_diff": dd[0], "acc_lo": dd[1], "acc_hi": dd[2],
                             "p_mcnemar": S.mcnemar_exact(cm, cr)})
    comp = pd.DataFrame(comp)
    comp["p_holm"] = S.holm(comp["p_mcnemar"].tolist())
    pd.DataFrame(rows).to_csv(out / "hint3.csv", index=False)
    comp.to_csv(out / "hint3_vs_readout.csv", index=False)
    log.info("HINT3 done")


SLOT = re.compile(r"\{\{[^}]*\}\}")


def norm(text):
    t = SLOT.sub(" slot ", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t)).strip()


def embed(texts, model_id="sentence-transformers/all-MiniLM-L6-v2",
          revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", bs=128):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    mod = AutoModel.from_pretrained(model_id, revision=revision).eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt")
            h = mod(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(v, dim=1).numpy())
    return np.vstack(out)


def bitext_overlap(out):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    train = read_jsonl(TASKS_DIR / "bitext" / "train.jsonl")
    test = read_jsonl(TASKS_DIR / "bitext" / "test.jsonl")
    tr, te = [norm(r["text"]) for r in train], [norm(r["text"]) for r in test]
    tr_lab = np.array([r["label"] for r in train])
    nn = {}
    for name, vec in [("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                      ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))]:
        Xtr = normalize(vec.fit_transform(tr)); Xte = normalize(vec.transform(te))
        sims = np.zeros(len(te)); arg = np.zeros(len(te), int)
        for i in range(0, len(te), 250):
            s = (Xte[i:i + 250] @ Xtr.T).toarray()
            sims[i:i + 250], arg[i:i + 250] = s.max(1), s.argmax(1)
        nn[name] = (sims, arg)
    Etr, Ete = embed(tr), embed(te)
    sims = np.zeros(len(te)); arg = np.zeros(len(te), int)
    for i in range(0, len(te), 250):
        s = Ete[i:i + 250] @ Etr.T
        sims[i:i + 250], arg[i:i + 250] = s.max(1), s.argmax(1)
    nn["sem"] = (sims, arg)
    te_lab = np.array([r["label"] for r in test])
    rows = []
    for name, (s, a) in nn.items():
        rows.append({"measure": name, "median_nn_sim": float(np.median(s)),
                     "share_ge_0.95": float(np.mean(s >= 0.95)), "share_ge_0.9": float(np.mean(s >= 0.9)),
                     "share_ge_0.8": float(np.mean(s >= 0.8)), "nn_same_label": float(np.mean(tr_lab[a] == te_lab)),
                     "share_with_slot": float(np.mean([bool(SLOT.search(r["text"])) for r in test]))})
    pd.DataFrame(rows).to_csv(out / "bitext_overlap.csv", index=False)
    pd.DataFrame({"id": [r["id"] for r in test], "nn_word": nn["word"][0], "nn_char": nn["char"][0],
                  "nn_sem": nn["sem"][0]}).to_csv(out / "bitext_overlap_items.csv", index=False)
    acc_rows = []
    ref = load("main", "qwen_readout", "bitext", "canonical")
    pos = {r["id"]: k for k, r in enumerate(test)}
    order = [pos[i] for i in ref["ids"]]
    for thr in (0.9, 0.8):
        far = (nn["sem"][0][order] < thr) & (nn["char"][0][order] < thr)
        for m in MAIN_SYSTEMS:
            d = load("main", m, "bitext", "canonical")
            c = d["P"].argmax(1) == d["y"]
            acc_rows.append({"threshold": thr, "model": m, "n_far": int(far.sum()), "acc_far": float(c[far].mean()),
                             "acc_near": float(c[~far].mean()), "acc_all": float(c.mean())})
    pd.DataFrame(acc_rows).to_csv(out / "bitext_overlap_acc.csv", index=False)
    log.info("bitext overlap: %s", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=1000)
    ap.add_argument("--ts-splits", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--only", default="main,backbone,hint3,overlap")
    args = ap.parse_args()
    out = results_dir("round1") / "tables"
    out.mkdir(parents=True, exist_ok=True)
    only = args.only.split(",")
    if "backbone" in only:
        backbone_checks(args.n_boot, out)
    if "hint3" in only:
        hint3(args.n_boot, out)
    if "main" in only:
        main_task_analyses(args.splits, args.ts_splits, out)
    if "overlap" in only:
        bitext_overlap(out)
    json.dump({"pplx_saved_temperature": PPLX_SAVED_T, "splits": args.splits, "ts_splits": args.ts_splits},
              open(out / "round1_offline_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
