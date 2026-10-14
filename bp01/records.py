from __future__ import annotations

import json
import random

from .config import OOS_LABEL, ROBUSTNESS_N, SEED, TASKS, TASKS_DIR, records_path
from .utils import get_logger, read_jsonl, write_jsonl

log = get_logger("records")

INSTRUCTIONS = {
    "banking77": "Which customer-service intent does this bank customer's message express?",
    "bitext": "Which customer-service intent does this customer's message express?",
    "clinc150": ("Which intent does this customer request express? "
                 "Choose 'oos' (out of scope) if none of the listed intents applies."),
    "clinc150_no_oos": "Which intent does this customer request express?",
}

OOS_DESCRIPTION = "Out of scope: the request does not match any of the other listed intents."


def humanize(label: str) -> str:
    s = label.replace("_", " ").replace("-", " ").strip()
    return s[:1].upper() + s[1:]


def criteria_for(labels: list[str]) -> dict[str, str]:
    return {lab: (OOS_DESCRIPTION if lab == OOS_LABEL else humanize(lab)) for lab in labels}


def robustness_subset(test: list[dict], task: str) -> list[dict]:
    rng = random.Random(f"{SEED}-{task}-subset")
    if len(test) <= ROBUSTNESS_N:
        return list(test)
    return sorted(rng.sample(test, ROBUSTNESS_N), key=lambda r: r["id"])


def make_record(item: dict, task: str, condition: str, options: list[str], instructions: str) -> dict:
    crit = criteria_for(options)
    return {
        "id": item["id"], "task": task, "condition": condition, "text": item["text"],
        "gold": item["label"], "instructions": instructions, "options": options,
        "criteria": {o: crit[o] for o in options},
    }


def build_task(task: str) -> dict[str, int]:
    labels = json.loads((TASKS_DIR / task / "labels.json").read_text())
    test = read_jsonl(TASKS_DIR / task / "test.jsonl")
    subset = robustness_subset(test, task)
    instr = INSTRUCTIONS[task]
    counts = {}

    def emit(cond, items, options, instructions=instr):
        recs = [make_record(it, task, cond, options, instructions) for it in items]
        write_jsonl(records_path(task, cond), recs)
        counts[cond] = len(recs)

    emit("canonical", test, labels)
    emit("reversed", subset, list(reversed(labels)))
    shuffled = list(labels)
    random.Random(f"{SEED}-{task}-shuffle").shuffle(shuffled)
    emit("shuffled", subset, shuffled)
    emit("repeat1", subset, labels)
    emit("repeat2", subset, labels)
    if task == "clinc150":
        in_scope = [l for l in labels if l != OOS_LABEL]
        emit("no_oos", test, in_scope, INSTRUCTIONS["clinc150_no_oos"])
    return counts


def main():
    summary = {t: build_task(t) for t in TASKS}
    for t, c in summary.items():
        log.info("%s: %s", t, c)

if __name__ == "__main__":
    main()
