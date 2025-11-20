"""SWE-bench mini runner: delegate to stock mini-swe-agent swebench script, enforce mini subset + HF sync."""

import json
import tarfile
from pathlib import Path

from pipeline.hf_sync import ensure_repo, upload_path
from pipeline.util import json_load, json_dump, load_conf

from minisweagent.run.extra import swebench as swe_run


MINI_DATASET = "MariusHobbhahn/SWE-bench-verified-mini"


def _rewrite_jsonl(preds_json: Path, jsonl_path: Path):
    preds = json_load(preds_json)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w") as w:
        for iid, rec in preds.items():
            w.write(
                json.dumps(
                    {
                        "instance_id": iid,
                        "model_name_or_path": rec.get("model_name_or_path", ""),
                        "model_patch": rec.get("model_patch", ""),
                        "status": rec.get("status", ""),
                    }
                )
                + "\n"
            )


def run():
    cfg = load_conf()
    run_id = cfg["run_id"]
    subset = cfg["swe"]["dataset_repo"]
    split = cfg["swe"]["split"]

    # Enforce mini subset only
    if "mini" not in subset.lower():
        raise ValueError("swe.dataset_repo must be the k-means mini subset (e.g., SWE-bench-verified-mini).")
    subset = MINI_DATASET

    out_dir = Path(f"runs/{run_id}/swe")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_repo = cfg["repos_fmt"]["sweb_dataset"]
    ensure_repo(ds_repo, "dataset")

    # Stock mini-swe-agent runner handles agent errors, trajectories, logging.
    swe_run.main(
        subset=subset,
        split=split,
        slice_spec="",
        filter_spec="",
        shuffle=False,
        output=str(out_dir),
        workers=1,
        model=None,
        model_class=None,
        redo_existing=False,
        config_spec=Path("conf/mini_qwen_thinking.yaml"),
        environment_class="singularity",
    )

    preds_path = out_dir / "preds.json"
    jsonl_path = out_dir / "all-preds.jsonl"
    if preds_path.exists():
        _rewrite_jsonl(preds_path, jsonl_path)
        # simple progress marker
        json_dump({"status": "done"}, out_dir / "progress.json")

        # Upload artifacts
        upload_path(ds_repo, str(preds_path), "dataset")
        upload_path(ds_repo, str(jsonl_path), "dataset")
        # include the minisweagent log and exit statuses if present
        log_file = out_dir / "minisweagent.log"
        for extra in [log_file] + list(out_dir.glob("exit_statuses_*.yaml")):
            if extra.exists():
                upload_path(ds_repo, str(extra), "dataset")

    # Final tarball with trajectories/logs
    tar_path = out_dir / "traces.tgz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname="swe")
    upload_path(ds_repo, str(tar_path), "dataset")


if __name__ == "__main__":
    run()
