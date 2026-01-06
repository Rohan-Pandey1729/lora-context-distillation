"""SWE-bench mini runner: delegate to stock mini-swe-agent swebench script, enforce mini subset + HF sync."""

import json
import os
import tarfile
import threading
from pathlib import Path

import yaml

from pipeline.hf_sync import ensure_repo, upload_path
from pipeline.util import json_load, json_dump, load_conf

from minisweagent.run.extra import swebench as swe_run


MINI_DATASET = "MariusHobbhahn/SWE-bench-verified-mini"
SYNC_INTERVAL_S = 60


def _collect_sync_paths(out_dir: Path) -> list[Path]:
    return [p for p in out_dir.rglob("*") if p.is_file()]


def _sync_once(repo_id: str, out_dir: Path, seen: dict[str, int]) -> None:
    for path in _collect_sync_paths(out_dir):
        if not path.exists() or path.is_dir():
            continue
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            continue
        key = str(path)
        if seen.get(key) == mtime:
            continue
        try:
            upload_path(repo_id, str(path), "dataset")
            seen[key] = mtime
        except Exception as e:
            print(f"[warn] sync failed for {path}: {e}")


def _sync_loop(repo_id: str, out_dir: Path, stop: threading.Event, seen: dict[str, int]) -> None:
    while not stop.is_set():
        try:
            _sync_once(repo_id, out_dir, seen)
        except Exception as e:
            print(f"[warn] sync loop error: {e}")
        stop.wait(SYNC_INTERVAL_S)


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


def _write_agent_config(run_id: str, port: int, registry_path: Path) -> Path:
    base_cfg_path = Path("conf/mini_qwen_thinking.yaml")
    cfg = yaml.safe_load(base_cfg_path.read_text())
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Invalid agent config in {base_cfg_path}")
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    model_kwargs = model_cfg.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        model_kwargs = {}
    model_kwargs["api_base"] = f"http://127.0.0.1:{port}/v1"
    model_cfg["model_kwargs"] = model_kwargs
    if registry_path.exists():
        model_cfg["litellm_model_registry"] = str(registry_path)
    cfg["model"] = model_cfg

    out_path = Path(f"runs/{run_id}/swe/agent_config.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_path


def _latest_exit_status(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("exit_statuses_*.yaml"), key=lambda p: p.stat().st_mtime_ns)
    return candidates[-1] if candidates else None


def _raise_on_exit_errors(out_dir: Path) -> None:
    exit_path = _latest_exit_status(out_dir)
    if not exit_path:
        return
    data = yaml.safe_load(exit_path.read_text()) or {}
    instances = data.get("instances_by_exit_status", {}) or {}
    if not isinstance(instances, dict):
        return
    bad = {
        status: insts
        for status, insts in instances.items()
        if isinstance(status, str)
        and any(word in status.lower() for word in ("error", "exception", "uncaught"))
    }
    if bad:
        total = sum(len(v) for v in bad.values() if isinstance(v, list))
        detail = ", ".join(f"{k}: {len(v)}" for k, v in bad.items())
        raise RuntimeError(f"SWE run had errors ({total} instances): {detail}")


def run():
    cfg = load_conf()
    run_id = cfg["run_id"]
    subset = cfg["swe"]["dataset_repo"]
    split = cfg["swe"]["split"]
    port = int(os.environ.get("VLLM_PORT", cfg["ports"]["vllm"]))
    registry_path = Path("conf/litellm_model_registry.json").resolve()

    # Enforce mini subset only
    if "mini" not in subset.lower():
        raise ValueError("swe.dataset_repo must be the k-means mini subset (e.g., SWE-bench-verified-mini).")
    subset = MINI_DATASET

    out_dir = Path(f"runs/{run_id}/swe")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_repo = cfg["repos_fmt"]["sweb_dataset"]
    ensure_repo(ds_repo, "dataset")

    stop = threading.Event()
    seen: dict[str, int] = {}
    t = threading.Thread(target=_sync_loop, args=(ds_repo, out_dir, stop, seen), daemon=True)
    t.start()

    try:
        # Stock mini-swe-agent runner handles agent errors, trajectories, logging.
        agent_cfg = _write_agent_config(run_id, port, registry_path)
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
            config_spec=agent_cfg,
            environment_class="singularity",
        )
    finally:
        stop.set()
        t.join(timeout=SYNC_INTERVAL_S)
        _sync_once(ds_repo, out_dir, seen)

    preds_path = out_dir / "preds.json"
    jsonl_path = out_dir / "all-preds.jsonl"
    _raise_on_exit_errors(out_dir)
    if not preds_path.exists():
        raise RuntimeError(f"Missing preds.json at {preds_path}")
    if not json_load(preds_path):
        raise RuntimeError(f"Empty preds.json at {preds_path}")
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
