"""Minimal SWE-bench mini runner that mirrors mini-swe-agent defaults, defaults to singularity, and fails fast."""

import json
import tarfile
from pathlib import Path

import yaml
from datasets import load_dataset

from pipeline.hf_sync import ensure_repo, maybe_download_file, maybe_download_json, upload_path
from pipeline.util import json_dump, json_load, load_conf

# mini-swe-agent import with fallback name
try:
    from mini_swe_agent.agents import DefaultAgent
    from mini_swe_agent.envs import get_environment
    from mini_swe_agent.models import get_model
except Exception:
    from minisweagent.agents import DefaultAgent
    from minisweagent.envs import get_environment
    from minisweagent.models import get_model

MINI_DATASET = "MariusHobbhahn/SWE-bench-verified-mini"


def _docker_image_from_instance(iid: str) -> str:
    """Same naming as upstream; always wrapped in docker:// for singularity."""
    docker_safe = iid.replace("__", "_1776_").lower()
    return f"docker://docker.io/swebench/sweb.eval.x86_64.{docker_safe}:latest"


def _load_existing_jsonl_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                iid = obj.get("instance_id")
                if iid is not None:
                    ids.add(iid)
            except Exception:
                continue
    return ids


def _rewrite_jsonl_from_preds(preds: dict, jsonl_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w") as w:
        for iid, rec in preds.items():
            w.write(
                json.dumps(
                    {
                        "instance_id": iid,
                        "model_name_or_path": rec.get("model_name_or_path", ""),
                        "model_patch": rec.get("model_patch", ""),
                    }
                )
                + "\n"
            )


def run():
    cfg = load_conf()
    run_id = cfg["run_id"]
    split = cfg["swe"]["split"]

    # Restrict to the k-means mini subset; fail fast otherwise.
    ds_name = cfg["swe"]["dataset_repo"]
    if "mini" not in ds_name.lower():
        raise ValueError("swe.dataset_repo must be the k-means mini subset (e.g., SWE-bench-verified-mini).")
    ds_name = MINI_DATASET

    local_out = Path(f"runs/{run_id}/swe")
    local_out.mkdir(parents=True, exist_ok=True)
    preds_path = local_out / "preds.json"
    progress_path = local_out / "progress.json"
    jsonl_path = local_out / "all-preds.jsonl"

    ds_repo = cfg["repos_fmt"]["sweb_dataset"]
    ensure_repo(ds_repo, "dataset")

    # Resume artifacts if present
    if not preds_path.exists():
        maybe_download_json(ds_repo, "preds.json", "dataset", str(preds_path))
    if not progress_path.exists():
        maybe_download_json(ds_repo, "progress.json", "dataset", str(progress_path))
    if not jsonl_path.exists():
        maybe_download_file(ds_repo, "all-preds.jsonl", "dataset", str(jsonl_path))

    preds = json_load(preds_path)
    done = set(preds.keys())

    if not jsonl_path.exists():
        _rewrite_jsonl_from_preds(preds, jsonl_path)
    else:
        jsonl_ids = _load_existing_jsonl_ids(jsonl_path)
        if not done.issubset(jsonl_ids):
            _rewrite_jsonl_from_preds(preds, jsonl_path)

    with open("conf/mini_qwen_thinking.yaml", "r") as f:
        mini_cfg = yaml.safe_load(f)

    # Force singularity; dataset supplies docker image names.
    env_cfg_base = dict(mini_cfg.get("environment", {}))
    env_cfg_base["environment_class"] = "singularity"

    model = get_model(config=mini_cfg.get("model", {}))
    ds = list(load_dataset(ds_name, split=split))

    print(f"[swe] {len(ds)} instances from {ds_name} ({split}); output -> {local_out}")

    jsonl_ids = _load_existing_jsonl_ids(jsonl_path)
    for idx, row in enumerate(ds, 1):
        iid = row["instance_id"]
        if iid in done:
            continue
        print(f"[swe] ({idx}/{len(ds)}) {iid}")

        env_cfg = dict(env_cfg_base)
        image = row.get("image_name") or row.get("image") or _docker_image_from_instance(iid)
        env_cfg["image"] = image if image.startswith("docker://") else f"docker://{image}"
        env = get_environment(env_cfg)

        try:
            agent = DefaultAgent(model, env, **mini_cfg.get("agent", {}))
            status, result = agent.run(row["problem_statement"])
        except Exception as exc:  # fail fast & verbose
            env.cleanup()
            raise RuntimeError(f"Agent failed on {iid}: {exc}") from exc
        finally:
            try:
                env.cleanup()
            except Exception:
                pass

        preds[iid] = {
            "model_name_or_path": mini_cfg["model"]["model_name"],
            "instance_id": iid,
            "model_patch": result,
        }
        json_dump(preds, str(preds_path))
        json_dump({"last_iid": iid, "status": status}, str(progress_path))

        if iid not in jsonl_ids:
            with jsonl_path.open("a") as w:
                w.write(
                    json.dumps(
                        {
                            "instance_id": iid,
                            "model_name_or_path": mini_cfg["model"]["model_name"],
                            "model_patch": result,
                        }
                    )
                    + "\n"
                )
            jsonl_ids.add(iid)

        upload_path(ds_repo, str(preds_path), "dataset")
        upload_path(ds_repo, str(progress_path), "dataset")
        upload_path(ds_repo, str(jsonl_path), "dataset")

    tar_path = local_out / "traces.tgz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(local_out, arcname="swe")
    upload_path(ds_repo, str(tar_path), "dataset")


if __name__ == "__main__":
    run()
