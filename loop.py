#!/usr/bin/env python3
"""Single-entry orchestrator for the Qwen3 loop."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path
import shutil

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_secret_env(var: str, secret_path: Path) -> None:
    if os.environ.get(var):
        return
    if secret_path.exists():
        os.environ[var] = secret_path.read_text().strip()


def _within_cwd(path: Path, root: Path) -> Path | None:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _set_local_dir_env(key: str, path: Path, root: Path) -> Path:
    fallback = _within_cwd(path, root) or root
    if key in os.environ:
        raw = Path(os.environ[key]).expanduser()
        target = _within_cwd(raw, root)
        if target is None:
            print(
                f"[warn] {key} escapes cwd ({raw}); resetting to {fallback}",
                file=sys.stderr,
            )
            target = fallback
            os.environ[key] = str(target)
    else:
        target = fallback
        os.environ[key] = str(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_local_runtime_env() -> None:
    root = Path.cwd().resolve()
    cache_root = root / ".cache"
    apptainer_root = cache_root / "apptainer"
    apptainer_tmp = apptainer_root / "tmp"

    _set_local_dir_env("BASE_CACHE_DIR", cache_root, root)
    _set_local_dir_env("XDG_CACHE_HOME", cache_root, root)
    _set_local_dir_env("TRITON_CACHE_DIR", cache_root / "triton", root)
    _set_local_dir_env("TORCH_HOME", cache_root / "torch", root)
    _set_local_dir_env("TORCH_EXTENSIONS_DIR", cache_root / "torch_extensions", root)
    _set_local_dir_env("TORCHINDUCTOR_CACHE_DIR", cache_root / "inductor", root)
    _set_local_dir_env("HF_HOME", cache_root / "hf", root)
    _set_local_dir_env("VLLM_CACHE_DIR", cache_root / "vllm", root)
    _set_local_dir_env("VLLM_CACHE_ROOT", cache_root / "vllm", root)

    _set_local_dir_env("UV_PROJECT_ENVIRONMENT", root / ".venv", root)
    _set_local_dir_env("UV_CACHE_DIR", cache_root / "uv", root)
    _set_local_dir_env("UV_AUTH_DIR", root / "share/uv", root)
    _set_local_dir_env("UV_PYTHON_INSTALL_DIR", root / "bin/uvpython", root)

    _set_local_dir_env("APPTAINER_CACHEDIR", apptainer_root / "cache", root)
    tmp_dir = _set_local_dir_env("APPTAINER_TMPDIR", apptainer_tmp, root)
    _set_local_dir_env("TMPDIR", tmp_dir, root)
    _set_local_dir_env("TEMP", tmp_dir, root)
    _set_local_dir_env("TMP", tmp_dir, root)
    tempfile.tempdir = str(tmp_dir)

    os.environ.setdefault("VLLM_USAGE_STATS", "0")


def load_conf() -> dict:
    ensure_local_runtime_env()
    _load_secret_env("GH_TOKEN", Path("secrets/gh_token"))
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Missing config at {CONFIG_PATH}")
    cfg = yaml.safe_load(_read_text(CONFIG_PATH))
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Invalid config format in {CONFIG_PATH}")
    run_id = os.environ.get("RUN_ID")
    if not run_id:
        slurm_id = os.environ.get("SLURM_JOB_ID")
        run_id = f"qwen3-{slurm_id}" if slurm_id else "manual"
    user = cfg.get("hf_username")
    if not user:
        raise RuntimeError("hf_username is required in config.yaml")
    repos = {k: v.format(user=user, run_id=run_id) for k, v in cfg.get("repos", {}).items()}
    cfg["run_id"] = run_id
    cfg["repos_fmt"] = repos
    return cfg


def json_load(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(obj: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(p)


def wait_vllm_ready(port: int, timeout_s: int = 900) -> None:
    import requests

    base = f"http://127.0.0.1:{port}"
    sess = requests.Session()
    sess.trust_env = False

    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            r = sess.get(f"{base}/health", timeout=5)
            if r.status_code == 200:
                r2 = sess.get(f"{base}/v1/models", timeout=10)
                if r2.status_code == 200:
                    return
        except Exception as exc:
            last = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"vLLM not ready after {timeout_s}s, last={last}")


def _pids_for_port(port: int) -> set[str]:
    pids: set[str] = set()
    for cmd in (
        ["lsof", "-ti", f":{port}"],
        ["bash", "-lc", f"ss -lptn 'sport = :{port}'"],
    ):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        except Exception:
            continue
        for match in re.findall(rb"\b\d+\b", out):
            pids.add(match.decode())
    return pids


def kill_port(port: int) -> None:
    for pid in _pids_for_port(port):
        try:
            subprocess.run(["kill", "-9", pid], check=False)
        except Exception:
            pass


def _can_bind(port: int) -> bool:
    sock = socket.socket()
    try:
        sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def pick_port(preferred: int | None = None) -> int:
    if preferred is not None and _can_bind(preferred):
        return preferred
    sock = socket.socket()
    sock.bind(("0.0.0.0", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def resolve_port(cfg: dict) -> int:
    env_port = os.environ.get("VLLM_PORT")
    if env_port:
        port = int(env_port)
        if not _can_bind(port):
            raise RuntimeError(f"VLLM_PORT {port} is already in use")
        return port
    preferred = int(cfg.get("ports", {}).get("vllm", 8000))
    if _can_bind(preferred):
        return preferred
    return pick_port(None)


def hf_api():
    from huggingface_hub import HfApi

    tok = os.environ.get("HF_TOKEN") or (
        Path("secrets/hf_token").read_text().strip() if Path("secrets/hf_token").exists() else ""
    )
    if not tok:
        raise RuntimeError("HF_TOKEN is required for HuggingFace Hub operations")
    return HfApi(token=tok)


def ensure_repo(repo_id: str, repo_type: str) -> None:
    from huggingface_hub import create_repo

    api = hf_api()
    create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=False, token=api.token)


def upload_path(repo_id: str, local_path: str, repo_type: str) -> None:
    from huggingface_hub import upload_file, upload_folder

    api = hf_api()
    if os.path.isdir(local_path):
        upload_folder(
            path=local_path,
            repo_id=repo_id,
            repo_type=repo_type,
            token=api.token,
            ignore_patterns=[".git/*", "**/*.pt", "**/*.bin"],
        )
    else:
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=os.path.basename(local_path),
            repo_id=repo_id,
            repo_type=repo_type,
            token=api.token,
        )


def list_latest_checkpoint(repo_id: str) -> str | None:
    from huggingface_hub import list_repo_files

    api = hf_api()
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="model", token=api.token)
    except Exception:
        return None
    prefixes = {f.split("/")[0] for f in files if f.split("/")[0].startswith("checkpoint-")}
    if not prefixes:
        return None
    def _ckpt_key(name: str) -> int:
        nums = re.findall(r"\d+", name)
        return int(nums[0]) if nums else -1
    return sorted(prefixes, key=_ckpt_key)[-1]


def download_folder_prefix(repo_id: str, prefix: str, local_dir: str) -> None:
    from huggingface_hub import hf_hub_download, list_repo_files

    api = hf_api()
    files = list_repo_files(repo_id=repo_id, repo_type="model", token=api.token)
    wanted = [f for f in files if f.startswith(prefix + "/")]
    for f in wanted:
        dst = os.path.join(local_dir, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        fp = hf_hub_download(
            repo_id=repo_id,
            filename=f,
            repo_type="model",
            token=api.token,
            local_dir=os.path.dirname(dst),
            local_dir_use_symlinks=False,
        )
        os.replace(fp, dst)


def _write_agent_config(run_id: str, port: int) -> Path:
    base_cfg_path = CONFIG_DIR / "mini_qwen_thinking.yaml"
    cfg = yaml.safe_load(_read_text(base_cfg_path))
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Invalid agent config in {base_cfg_path}")
    env_cfg = cfg.get("environment", {})
    if isinstance(env_cfg, dict):
        # Keep environment_class as "singularity" so swebench injects the image field.
        if env_cfg.get("environment_class") == "swe_singularity_env.SingularityEnvironment":
            env_cfg["environment_class"] = "singularity"
        cfg["environment"] = env_cfg
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    model_kwargs = model_cfg.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        model_kwargs = {}
    model_kwargs["api_base"] = f"http://127.0.0.1:{port}/v1"
    model_cfg["model_kwargs"] = model_kwargs

    registry = os.environ.get("LITELLM_MODEL_REGISTRY_PATH")
    registry_path = Path(registry) if registry else (CONFIG_DIR / "litellm_model_registry.json")
    if not registry_path.exists():
        raise RuntimeError(f"Missing LiteLLM model registry at {registry_path}")
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
    data = yaml.safe_load(_read_text(exit_path)) or {}
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


def _rewrite_jsonl(preds_json: Path, jsonl_path: Path) -> None:
    preds = json_load(preds_json)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as w:
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


def run_swe(cfg: dict, port: int) -> None:
    from minisweagent.run.extra import swebench as swe_run
    from minisweagent import environments as mswe_envs

    # Ensure swebench sets the image for "singularity" while we provide a custom implementation.
    mswe_envs._ENVIRONMENT_MAPPING["singularity"] = "swe_singularity_env.SingularityEnvironment"

    swe_cfg = cfg.get("swe", {}) or {}
    subset = swe_cfg.get("dataset_repo")
    split = swe_cfg.get("split")
    slice_spec = swe_cfg.get("slice_spec") or "0:10"
    filter_spec = swe_cfg.get("filter_spec") or ""
    if not subset or not split:
        raise RuntimeError("swe.dataset_repo and swe.split are required")

    if "mini" not in subset.lower():
        raise ValueError("swe.dataset_repo must be the k-means mini subset (e.g., SWE-bench-verified-mini).")
    subset = "MariusHobbhahn/SWE-bench-verified-mini"

    run_id = cfg["run_id"]
    out_dir = Path(f"runs/{run_id}/swe")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_repo = cfg["repos_fmt"]["sweb_dataset"]
    ensure_repo(ds_repo, "dataset")

    stop = threading.Event()
    seen: dict[str, int] = {}

    def _collect_sync_paths() -> list[Path]:
        return [p for p in out_dir.rglob("*") if p.is_file()]

    def _sync_once() -> None:
        for path in _collect_sync_paths():
            try:
                mtime = path.stat().st_mtime_ns
            except FileNotFoundError:
                continue
            key = str(path)
            if seen.get(key) == mtime:
                continue
            try:
                upload_path(ds_repo, str(path), "dataset")
                seen[key] = mtime
            except Exception as exc:
                print(f"[warn] sync failed for {path}: {exc}")

    def _sync_loop() -> None:
        while not stop.is_set():
            try:
                _sync_once()
            except Exception as exc:
                print(f"[warn] sync loop error: {exc}")
            stop.wait(60)

    t = threading.Thread(target=_sync_loop, daemon=True)
    t.start()

    try:
        agent_cfg = _write_agent_config(run_id, port)
        swe_run.main(
            subset=subset,
            split=split,
            slice_spec=slice_spec,
            filter_spec=filter_spec,
            shuffle=False,
            output=str(out_dir),
            workers=1,
            model=None,
            model_class=None,
            redo_existing=False,
            config_spec=agent_cfg,
            environment_class=None,
        )
    finally:
        stop.set()
        t.join(timeout=60)
        _sync_once()

    preds_path = out_dir / "preds.json"
    jsonl_path = out_dir / "all-preds.jsonl"
    _raise_on_exit_errors(out_dir)
    if not preds_path.exists():
        raise RuntimeError(f"Missing preds.json at {preds_path}")
    if not json_load(preds_path):
        raise RuntimeError(f"Empty preds.json at {preds_path}")

    _rewrite_jsonl(preds_path, jsonl_path)
    json_dump({"status": "done"}, out_dir / "progress.json")

    upload_path(ds_repo, str(preds_path), "dataset")
    upload_path(ds_repo, str(jsonl_path), "dataset")
    for extra in [out_dir / "minisweagent.log"] + list(out_dir.glob("exit_statuses_*.yaml")):
        if extra.exists():
            upload_path(ds_repo, str(extra), "dataset")

    tar_path = out_dir / "traces.tgz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname="swe")
    upload_path(ds_repo, str(tar_path), "dataset")


def strip_thinking(cfg: dict, preds_json: Path, out_jsonl: Path, dataset_repo: str | None = None, split: str | None = None) -> None:
    import jsonlines
    from datasets import load_dataset

    dataset_repo = dataset_repo or cfg.get("swe", {}).get("dataset_repo")
    split = split or cfg.get("swe", {}).get("split")
    if not dataset_repo or not split:
        raise RuntimeError("swe.dataset_repo and swe.split are required")

    preds = json_load(preds_json)
    if not preds:
        raise RuntimeError(f"No predictions found in {preds_json}")

    ds = load_dataset(dataset_repo, split=split)
    prompts = {row["instance_id"]: row["problem_statement"] for row in ds}

    think_re = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

    def strip_think(s: str) -> str:
        s = re.sub(think_re, "", s)
        return s.replace("<think>", "").replace("</think>", "").strip()

    def resolve_instruction(iid: str, rec: dict) -> str:
        for key in ("prompt", "instruction", "problem_statement"):
            val = rec.get(key)
            if val:
                return val
        if iid in prompts:
            return prompts[iid]
        raise KeyError(f"Missing prompt for {iid}")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(out_jsonl, "w") as w:
        for iid, rec in preds.items():
            out = strip_think(rec.get("model_patch", ""))
            instruction = resolve_instruction(iid, rec)
            w.write({"id": iid, "instruction": instruction, "output": out})
    print(f"Wrote SFT JSONL to {out_jsonl}")


def train_unsloth_lora(cfg: dict) -> None:
    from datasets import load_dataset
    from huggingface_hub import create_repo, upload_folder
    from transformers import TrainingArguments, TrainerCallback
    from trl import SFTTrainer
    from unsloth import FastModel

    try:
        import torch
    except Exception as exc:
        print(f"[fatal] PyTorch import failed: {exc}", file=sys.stderr)
        sys.exit(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")

    run_id = cfg["run_id"]
    repos = cfg["repos_fmt"]

    data_jsonl = Path(f"runs/{run_id}/sft/sft_qwenA_from_B_mini.jsonl")
    out_dir = Path(f"runs/{run_id}/trainA")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_jsonl.exists() or data_jsonl.stat().st_size == 0:
        raise RuntimeError(f"SFT dataset missing or empty at {data_jsonl}")

    latest = list_latest_checkpoint(repos["a_ckpt_model"])
    if latest and not any(p.name.startswith("checkpoint-") for p in out_dir.glob("checkpoint-*")):
        download_folder_prefix(repos["a_ckpt_model"], latest, str(out_dir))

    ds = load_dataset("json", data_files=str(data_jsonl), split="train")

    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg["model_a_base"],
        load_in_4bit=True,
        max_seq_length=cfg["train"]["max_len"],
        dtype=None,
        device_map="auto",
    )
    model = FastModel.get_peft_model(
        model,
        r=1,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    def fmt(ex):
        return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"

    training_args = TrainingArguments(
        per_device_train_batch_size=cfg["train"]["bsz"],
        gradient_accumulation_steps=cfg["train"]["grad_acc"],
        learning_rate=cfg["train"]["lr"],
        lr_scheduler_type="cosine",
        num_train_epochs=cfg["train"]["epochs"],
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=5,
        save_steps=cfg["train"]["save_steps"],
        output_dir=str(out_dir),
        logging_dir=str(out_dir / "logs"),
        report_to=[],
        optim="adamw_8bit",
        dataloader_num_workers=2,
    )

    def _push_folder(repo_id: str, local_path: Path) -> None:
        api = hf_api()
        create_repo(repo_id, repo_type="model", exist_ok=True, private=False, token=api.token)
        upload_folder(
            path=str(local_path),
            repo_id=repo_id,
            repo_type="model",
            token=api.token,
            ignore_patterns=["*.bin", "*.pt", ".git/*"],
        )

    class PushOnSave(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            last_ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if last_ckpt.is_dir():
                _push_folder(repos["a_ckpt_model"], last_ckpt)
            try:
                merged = Path(args.output_dir) / f"merged_step_{state.global_step}"
                merged.mkdir(parents=True, exist_ok=True)
                FastModel.push_to_hub_merged(
                    model,
                    tokenizer,
                    save_directory=str(merged),
                    repo_id=None,
                    token=os.environ["HF_TOKEN"],
                    push_to_hub=False,
                )
                _push_folder(repos["a_merged_model"], merged)
            except Exception as exc:
                print(f"[warn] merge snapshot failed: {exc}")
            return control

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        formatting_func=fmt,
        max_seq_length=cfg["train"]["max_len"],
        packing=True,
        args=training_args,
    )

    trainer.add_callback(PushOnSave())
    resume = any(p.name.startswith("checkpoint-") for p in out_dir.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume)

    final_dir = out_dir / "final_merged_A"
    final_dir.mkdir(parents=True, exist_ok=True)
    FastModel.push_to_hub_merged(
        model,
        tokenizer,
        save_directory=str(final_dir),
        repo_id=None,
        token=os.environ["HF_TOKEN"],
        push_to_hub=False,
    )
    _push_folder(repos["a_merged_model"], final_dir)

    if (ROOT / "runs/A_final").is_symlink():
        (ROOT / "runs/A_final").unlink()
    (ROOT / "runs").mkdir(exist_ok=True)
    os.system(f"ln -sfn {final_dir} runs/A_final")


def apply_diff_linear(cfg: dict) -> None:
    run_id = cfg["run_id"]
    new_a = os.path.abspath("runs/A_final")
    if not os.path.isdir(new_a):
        raise RuntimeError("runs/A_final missing")

    mk_path = CONFIG_DIR / "mk_apply_template.yml"
    mk = yaml.safe_load(_read_text(mk_path))
    for model in mk.get("models", []):
        if model.get("model") == "__NEW_A__":
            model["model"] = new_a

    with tempfile.TemporaryDirectory() as td:
        yml_path = Path(td) / "mk.yml"
        yml_path.write_text(yaml.safe_dump(mk, sort_keys=False))
        out = Path(f"runs/{run_id}/B_new")
        out.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["mergekit-yaml", str(yml_path), str(out)])
    print("New B saved at", out)


def upload_new_b(cfg: dict) -> None:
    run_id = cfg["run_id"]
    repo = cfg["repos_fmt"]["b_new_model"]
    ensure_repo(repo, "model")
    upload_path(repo, f"runs/{run_id}/B_new", "model")


def snap_logs(cfg: dict) -> None:
    run_id = cfg["run_id"]
    src = []
    if Path("logs").is_dir():
        src.append("logs")
    train_logs = Path(f"runs/{run_id}/trainA/logs")
    if train_logs.is_dir():
        src.append(str(train_logs))
    if not src:
        return
    out_tar = Path(f"runs/{run_id}/meta/logs-{int(time.time())}.tar.gz")
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar, "w:gz") as tar:
        for src_path in src:
            src_path = os.path.abspath(src_path)
            for root, _, files in os.walk(src_path):
                for f in files:
                    full = os.path.join(root, f)
                    arc = os.path.relpath(full, start=os.path.dirname(src_path))
                    tar.add(full, arcname=os.path.join(os.path.basename(src_path), os.path.relpath(full, src_path)))
    repo = cfg["repos_fmt"].get("logs_dataset")
    if repo:
        ensure_repo(repo, "dataset")
        upload_path(repo, str(out_tar), "dataset")


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        check=check,
        text=True,
        capture_output=True,
        cwd=ROOT,
    )


def _git_has_changes() -> bool:
    res = _git("status", "--porcelain=v1", check=False)
    return bool(res.stdout.strip())


def _git_current_branch() -> str | None:
    res = _git("rev-parse", "--abbrev-ref", "HEAD", check=False)
    branch = res.stdout.strip()
    if res.returncode != 0 or not branch or branch == "HEAD":
        return None
    return branch


def _git_default_branch() -> str:
    res = _git("symbolic-ref", "refs/remotes/origin/HEAD", check=False)
    if res.returncode == 0:
        ref = res.stdout.strip()
        if ref.startswith("refs/remotes/origin/"):
            return ref.rsplit("/", 1)[-1]
    current = _git_current_branch()
    return current or "main"


def commit_and_pr(cfg: dict) -> None:
    if not _git_has_changes():
        return

    run_id = cfg["run_id"]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    branch = f"run/{run_id}-{stamp}-{suffix}"
    title = f"Run {run_id} logs {stamp}"
    body = (
        f"Automated run artifacts for `{run_id}`.\n\n"
        f"- Logs: `logs/`\n"
        f"- Run outputs: `runs/{run_id}/`\n"
    )

    try:
        _git("switch", "-c", branch)
    except subprocess.CalledProcessError:
        _git("checkout", "-b", branch)
    _git("add", "-A")
    staged = _git("diff", "--cached", "--quiet", check=False)
    if staged.returncode == 0:
        return
    _git("commit", "-m", title)
    try:
        _git("push", "-u", "origin", branch)
    except subprocess.CalledProcessError as exc:
        print(f"[warn] git push failed; skipping PR: {exc.stderr.strip()}")
        return
    if shutil.which("gh") is None:
        print("[warn] gh not found; skipping PR creation")
        return
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        auth = subprocess.run(
            ["gh", "auth", "status", "-h", "github.com"],
            check=False,
            text=True,
            capture_output=True,
            cwd=ROOT,
        )
        if auth.returncode != 0:
            print("[warn] gh not authenticated; skipping PR creation")
            return
    base = _git_default_branch()
    try:
        subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--base",
                base,
                "--head",
                branch,
                "--title",
                title,
                "--body",
                body,
            ],
            check=True,
            text=True,
            capture_output=True,
            cwd=ROOT,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[warn] gh pr create failed: {exc.stderr.strip()}")


def start_vllm(model: str, port: int, tp: int, logdir: Path) -> subprocess.Popen:
    logdir.mkdir(parents=True, exist_ok=True)
    stdout = (logdir / "server.stdout.log").open("w", encoding="utf-8")
    stderr = (logdir / "server.stderr.log").open("w", encoding="utf-8")
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--model",
        model,
        "--dtype",
        "auto",
        "--quantization",
        "fp8",
        "--tensor-parallel-size",
        str(tp),
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.92",
        "--enforce-eager",
        "--max-num-seqs",
        "64",
        "--disable-log-requests",
    ]
    return subprocess.Popen(cmd, stdout=stdout, stderr=stderr)


def full_run() -> None:
    cfg = load_conf()
    run_id = cfg["run_id"]
    port = resolve_port(cfg)
    model_b = cfg["model_b_base"]
    logdir = Path(f"logs/vllm_{run_id}")

    kill_port(port)
    proc = start_vllm(model_b, port, tp=2, logdir=logdir)
    try:
        wait_vllm_ready(port, timeout_s=2400)
        run_swe(cfg, port)
        preds_json = Path(f"runs/{run_id}/swe/preds.json")
        out_jsonl = Path(f"runs/{run_id}/sft/sft_qwenA_from_B_mini.jsonl")
        strip_thinking(cfg, preds_json, out_jsonl)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        kill_port(port)

    train_unsloth_lora(cfg)
    apply_diff_linear(cfg)
    upload_new_b(cfg)
    try:
        snap_logs(cfg)
    except Exception as exc:
        print(f"[warn] log snapshot failed: {exc}")
    try:
        commit_and_pr(cfg)
    except Exception as exc:
        print(f"[warn] git/pr automation failed: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("pick-port")
    sub.add_parser("full")
    sub.add_parser("swe")
    sub.add_parser("strip")
    sub.add_parser("train")
    sub.add_parser("apply-diff")
    sub.add_parser("upload-b")
    sub.add_parser("snap-logs")

    args = ap.parse_args()
    cfg = load_conf()

    if args.cmd == "pick-port":
        port = resolve_port(cfg)
        print(port)
        return
    if args.cmd == "full":
        full_run()
        return

    if args.cmd == "swe":
        port = resolve_port(cfg)
        run_swe(cfg, port)
        return
    if args.cmd == "strip":
        run_id = cfg["run_id"]
        strip_thinking(
            cfg,
            Path(f"runs/{run_id}/swe/preds.json"),
            Path(f"runs/{run_id}/sft/sft_qwenA_from_B_mini.jsonl"),
        )
        return
    if args.cmd == "train":
        train_unsloth_lora(cfg)
        return
    if args.cmd == "apply-diff":
        apply_diff_linear(cfg)
        return
    if args.cmd == "upload-b":
        upload_new_b(cfg)
        return
    if args.cmd == "snap-logs":
        snap_logs(cfg)
        return

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
