from __future__ import annotations

import copy
import math

from ..config import HF_MODELS, HF_REVISIONS, READOUT_PRUNE, READOUT_TOP_LOGPROBS
from ..prompts import READOUT_PREFILL, readout_user_prompt
from ..readout import CLOSE, DIGITS, _prefix_sets, char_token_ids, label_strings
from ..utils import Timer, append_jsonl, get_logger
from .common import base_parser, jobs, pred_row

log = get_logger("exact")


def load(llm_key: str, device: str):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    repo, rev = HF_MODELS[llm_key], HF_REVISIONS.get(llm_key)
    cfg = AutoConfig.from_pretrained(repo, revision=rev)
    arch = (cfg.architectures or [""])[0]
    if "ConditionalGeneration" in arch:
        from transformers import AutoModelForImageTextToText as cls
    else:
        cls = AutoModelForCausalLM
    dtype = "auto" if device == "cuda" else torch.float32
    model = cls.from_pretrained(repo, revision=rev, dtype=dtype, device_map=device).eval()
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    log.info("loaded %s@%s as %s (%s)", repo, rev, cls.__name__, device)
    return model, tok


def chat_ids(tok, user_text: str) -> list[int]:
    text = tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return tok.encode(text + READOUT_PREFILL, add_special_tokens=False)


def score_record(model, ids: list[int], k: int, cids: dict[str, int], device: str):
    import torch

    labels, prefixes = _prefix_sets(k)
    cand = list(DIGITS + CLOSE)
    cand_ids = torch.tensor([cids[c] for c in cand], device=device)
    nodes: dict[str, tuple[dict[str, float], float]] = {}

    def record(s, logits):
        lp = torch.log_softmax(logits.float(), dim=-1)
        kth = torch.topk(lp, READOUT_TOP_LOGPROBS).values[-1].item()
        vals = lp[cand_ids].tolist()
        nodes[s] = (dict(zip(cand, vals)), kth)

    def expand(s, cache, pos):
        for d in DIGITS:
            child = s + d
            if child not in prefixes:
                continue
            c2 = copy.deepcopy(cache)
            with torch.inference_mode():
                out = model(input_ids=torch.tensor([[cids[d]]], device=device), past_key_values=c2, use_cache=True,
                            cache_position=torch.tensor([pos], device=device))
            record(child, out.logits[0, -1])
            if any(lab.startswith(child) and lab != child for lab in labels):
                expand(child, out.past_key_values, pos + 1)
            del c2, out

    with torch.inference_mode():
        out = model(input_ids=torch.tensor([ids], device=device), use_cache=True,
                    cache_position=torch.arange(len(ids), device=device))
    record("", out.logits[0, -1])
    expand("", out.past_key_values, len(ids))
    return nodes


def exact_probs(nodes, k):
    lps = {}
    for lab in label_strings(k):
        lp = sum(nodes[lab[:j]][0][lab[j]] for j in range(len(lab))) + nodes[lab][0][CLOSE]
        lps[lab] = lp
    raw = [math.exp(lps[lab]) for lab in label_strings(k)]
    mass = sum(raw)
    return [r / mass for r in raw], mass


def approx_probs(nodes, k, prune=READOUT_PRUNE):
    labels, prefixes = _prefix_sets(k)
    label_lp, frontier = {}, [("", 0.0)]
    while frontier:
        nxt = []
        for s, lp in frontier:
            cand, kth = nodes[s]
            if s in labels and cand[CLOSE] >= kth:
                label_lp[s] = lp + cand[CLOSE]
            for d in DIGITS:
                child = s + d
                if child in prefixes and cand[d] >= kth and math.exp(lp + cand[d]) >= prune:
                    nxt.append((child, lp + cand[d]))
        frontier = nxt
    raw = [math.exp(label_lp.get(lab, float("-inf"))) for lab in label_strings(k)]
    mass = sum(raw)
    return ([r / mass for r in raw] if mass > 0 else [1.0 / k] * k), mass


def self_check(model, ids, cids, nodes, device):
    import torch
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([ids + [cids["1"]]], device=device))
    lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    diff = max(abs(lp[cids[c]].item() - nodes["1"][0][c]) for c in DIGITS + CLOSE)
    log.info("self-check (cache vs full forward, node '1'): max |d logprob| = %.2e", diff)
    return diff


def main():
    import torch

    ap = base_parser("Exact LLM readout without truncation or pruning (transformers)")
    ap.add_argument("--llm", choices=["qwen", "qwen_bf16", "mac_readout"], required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, tok = load(args.llm, device)
    cids = char_token_ids(tok)
    model_key = f"{args.llm}_exact"
    for task, cond, recs, out in jobs(args, model_key):
        checked = False
        for r in recs:
            k = len(r["options"])
            ids = chat_ids(tok, readout_user_prompt(r))
            with Timer() as t:
                nodes = score_record(model, ids, k, cids, device)
            p_ex, m_ex = exact_probs(nodes, k)
            p_ap, m_ap = approx_probs(nodes, k)
            meta = {"mass_exact": m_ex, "mass_approx": m_ap, "approx": p_ap, "nodes": len(nodes)}
            if not checked:
                meta["self_check_max_abs_dlogprob"] = self_check(model, ids, cids, nodes, device)
                checked = True
            append_jsonl(out, [pred_row(r, model_key, dict(zip(r["options"], p_ex)), t.seconds, len(ids), meta)])
        log.info("done %s", out)


if __name__ == "__main__":
    main()
