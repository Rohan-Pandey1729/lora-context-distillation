import os, tarfile, time, pathlib, shutil
from pipeline.util import load_conf
from pipeline.hf_sync import ensure_repo, upload_path

def _tar_folder(src_paths, out_tar, excludes=()):
    os.makedirs(os.path.dirname(out_tar), exist_ok=True)
    with tarfile.open(out_tar, "w:gz") as tar:
        for src in src_paths:
            src = os.path.abspath(src)
            if not os.path.exists(src):
                continue
            for root, dirs, files in os.walk(src):
                # apply excludes as substrings on full path
                skip = False
                for ex in excludes:
                    if ex and ex in root:
                        skip = True; break
                if skip: 
                    continue
                for f in files:
                    full = os.path.join(root, f)
                    if any(ex and ex in full for ex in excludes):
                        continue
                    arc = os.path.relpath(full, start=os.path.dirname(src))
                    tar.add(full, arcname=os.path.join(os.path.basename(src), os.path.relpath(full, src)))

def snap_code():
    cfg = load_conf()
    run_id = cfg["run_id"]
    # what counts as "the code" for replay: env, conf, slurm, bin, pipeline
    src = ["env", "conf", "slurm", "bin", "pipeline"]
    excludes = [".hf/", ".cache/", "runs/", "logs/", "results/", "secrets/hf_token"]
    out_tar = f"runs/{run_id}/meta/code-snapshot-{int(time.time())}.tar.gz"
    _tar_folder(src, out_tar, excludes)
    repo = cfg["repos_fmt"].get("code_dataset", f"{cfg[hf_username]}/qwen3-loop-code-{run_id}")
    ensure_repo(repo, "dataset")
    upload_path(repo, out_tar, "dataset")

def snap_logs():
    cfg = load_conf()
    run_id = cfg["run_id"]
    # include vLLM logs and trainer logs if present
    src = []
    if os.path.isdir("logs"): src.append("logs")
    ta = f"runs/{run_id}/trainA/logs"
    if os.path.isdir(ta): src.append(ta)
    if not src:
        return
    out_tar = f"runs/{run_id}/meta/logs-{int(time.time())}.tar.gz"
    _tar_folder(src, out_tar, excludes=[])
    repo = cfg["repos_fmt"].get("logs_dataset", f"{cfg[hf_username]}/qwen3-loop-logs-{run_id}")
    ensure_repo(repo, "dataset")
    upload_path(repo, out_tar, "dataset")

if __name__ == "__main__":
    # minimal CLI: python -m pipeline.snap_and_sync code|logs
    import sys
    cmd = sys.argv[1] if len(sys.argv)>1 else "code"
    if cmd == "code": snap_code()
    elif cmd == "logs": snap_logs()
    else: 
        print("usage: python -m pipeline.snap_and_sync [code|logs]")
