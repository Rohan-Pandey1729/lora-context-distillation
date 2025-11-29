import argparse
import os
import re

import jsonlines
from datasets import load_dataset

from pipeline.util import json_load, load_conf

THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def strip_think(s: str) -> str:
    """Drop <think> blocks from the model output while leaving the rest intact."""
    s = re.sub(THINK_RE, "", s)
    return s.replace("<think>", "").replace("</think>", "").strip()


def build_prompt_lookup(dataset_repo: str, split: str) -> dict[str, str]:
    """Load the SWEBench split and map instance_id -> original problem statement."""
    ds = load_dataset(dataset_repo, split=split)
    return {row["instance_id"]: row["problem_statement"] for row in ds}


def resolve_instruction(iid: str, rec: dict, prompts: dict[str, str]) -> str:
    """Prefer any prompt already present in preds; otherwise fall back to dataset prompt."""
    for key in ("prompt", "instruction", "problem_statement"):
        val = rec.get(key)
        if val:
            return val
    if iid in prompts:
        return prompts[iid]
    raise KeyError(f"Missing prompt for {iid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_json", required=True, help="mini-swe-agent preds.json")
    ap.add_argument("--out_jsonl", required=True, help="path for stripped SFT jsonl")
    ap.add_argument("--dataset_repo", default=None, help="SWEBench dataset repo (defaults to conf.swe.dataset_repo)")
    ap.add_argument("--split", default=None, help="dataset split (defaults to conf.swe.split)")
    args = ap.parse_args()

    cfg = load_conf()
    dataset_repo = args.dataset_repo or cfg["swe"]["dataset_repo"]
    split = args.split or cfg["swe"]["split"]

    preds = json_load(args.preds_json)
    prompts = build_prompt_lookup(dataset_repo, split)

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with jsonlines.open(args.out_jsonl, "w") as w:
        for iid, rec in preds.items():
            out = strip_think(rec.get("model_patch", ""))
            instruction = resolve_instruction(iid, rec, prompts)
            w.write({"id": iid, "instruction": instruction, "output": out})
    print(f"Wrote SFT JSONL to {args.out_jsonl}")


if __name__ == "__main__":
    main()
