import re, jsonlines, os, argparse
from pipeline.util import json_load
THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
def strip_think(s:str) -> str:
    s = re.sub(THINK_RE, "", s)
    return s.replace("<think>","").replace("</think>","").strip()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()
    preds = json_load(args.preds_json)
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with jsonlines.open(args.out_jsonl, "w") as w:
        for iid, rec in preds.items():
            out = strip_think(rec.get("model_patch",""))
            w.write({"id": iid, "instruction": f"Solve SWE-bench issue {iid}. Provide only the final patch diff or commands, no hidden thinking.", "output": out})
    print(f"Wrote SFT JSONL to {args.out_jsonl}")
if __name__ == "__main__":
    main()
