import os, json, tarfile

from datasets import load_dataset
import yaml

from pipeline.hf_sync import ensure_repo, upload_path, maybe_download_json, maybe_download_file
from pipeline.util import load_conf, json_load, json_dump

# mini-swe-agent import with fallback name
try:
    from mini_swe_agent.models import get_model
    from mini_swe_agent.envs import get_environment
    from mini_swe_agent.agents import DefaultAgent
except Exception:
    from minisweagent.models import get_model
    from minisweagent.envs import get_environment
    from minisweagent.agents import DefaultAgent

def _load_existing_jsonl_ids(path):
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                iid = obj.get("instance_id")
                if iid is not None:
                    ids.add(iid)
            except Exception:
                continue
    return ids

def _rewrite_jsonl_from_preds(preds, jsonl_path):
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w") as w:
        for iid, rec in preds.items():
            row = {
                "instance_id": iid,
                "model_name_or_path": rec.get("model_name_or_path",""),
                "model_patch": rec.get("model_patch",""),
            }
            w.write(json.dumps(row) + "\n")

def run():
    cfg = load_conf()
    run_id = cfg["run_id"]
    ds_name = cfg["swe"]["dataset_repo"]; split = cfg["swe"]["split"]
    local_out = f"runs/{run_id}/swe"
    os.makedirs(local_out, exist_ok=True)

    preds_path = f"{local_out}/preds.json"             # dict format
    progress_path = f"{local_out}/progress.json"
    jsonl_path = f"{local_out}/all-preds.jsonl"        # line-per-pred for your uploads

    ds_repo = cfg["repos_fmt"]["sweb_dataset"]
    ensure_repo(ds_repo, "dataset")

    # resume: pull latest remote artifacts if missing locally
    if not os.path.exists(preds_path):
        maybe_download_json(ds_repo, "preds.json", "dataset", preds_path)
    if not os.path.exists(progress_path):
        maybe_download_json(ds_repo, "progress.json", "dataset", progress_path)
    if not os.path.exists(jsonl_path):
        maybe_download_file(ds_repo, "all-preds.jsonl", "dataset", jsonl_path)

    preds = json_load(preds_path)
    done = set(preds.keys())

    # ensure JSONL contains current preds, no duplicates
    if not os.path.exists(jsonl_path):
        _rewrite_jsonl_from_preds(preds, jsonl_path)
    else:
        # if JSONL existed but is behind, rebuild from preds for idempotence
        jsonl_ids = _load_existing_jsonl_ids(jsonl_path)
        if not done.issubset(jsonl_ids):
            _rewrite_jsonl_from_preds(preds, jsonl_path)

    # model and dataset
    with open("conf/mini_qwen_thinking.yaml","r") as f:
        mini_cfg = yaml.safe_load(f)
    model = get_model(config=mini_cfg.get("model", {}))
    ds = load_dataset(ds_name, split=split)

    jsonl_ids = _load_existing_jsonl_ids(jsonl_path)
    for row in ds:
        iid = row["instance_id"]
        if iid in done:
            continue

        env_cfg = dict(mini_cfg.get("environment", {}))
        image = row.get("image_name") or row.get("image") or None
        if image:
            env_cfg["image"] = f"docker://{image}"
        env = get_environment(env_cfg)

        try:
            agent = DefaultAgent(model, env, **mini_cfg.get("agent", {}))
            status, result = agent.run(row["problem_statement"])
        finally:
            try:
                env.cleanup()
            except Exception:
                pass

        # update preds.json
        preds[iid] = {"model_name_or_path": mini_cfg["model"]["model_name"], "instance_id": iid, "model_patch": result}
        json_dump(preds, preds_path)
        json_dump({"last_iid": iid, "status": status}, progress_path)

        # append to all-preds.jsonl only if new
        if iid not in jsonl_ids:
            with open(jsonl_path, "a") as w:
                w.write(json.dumps({"instance_id": iid,
                                    "model_name_or_path": mini_cfg["model"]["model_name"],
                                    "model_patch": result}) + "\n")
            jsonl_ids.add(iid)

        # sync 3 artifacts every task
        upload_path(ds_repo, preds_path, "dataset")
        upload_path(ds_repo, progress_path, "dataset")
        upload_path(ds_repo, jsonl_path, "dataset")

    # final traces bundle
    tar_path = f"{local_out}/traces.tgz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(local_out, arcname="swe")
    upload_path(ds_repo, tar_path, "dataset")

if __name__ == "__main__":
    run()
