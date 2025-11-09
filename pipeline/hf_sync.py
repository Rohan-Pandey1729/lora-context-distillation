import os, pathlib
from typing import Optional
from huggingface_hub import HfApi, hf_hub_download, upload_file, upload_folder, create_repo, list_repo_files

def api():
    tok = os.environ.get("HF_TOKEN") or (pathlib.Path("secrets/hf_token").read_text().strip() if pathlib.Path("secrets/hf_token").exists() else None)
    return HfApi(token=tok)

def ensure_repo(repo_id: str, repo_type: str):
    a = api()
    create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=False, token=a.token)

def upload_path(repo_id: str, local_path: str, repo_type: str):
    a = api()
    if os.path.isdir(local_path):
        upload_folder(path=local_path, repo_id=repo_id, repo_type=repo_type, token=a.token, ignore_patterns=[".git/*","**/*.pt","**/*.bin"])
    else:
        upload_file(path_or_fileobj=local_path, path_in_repo=os.path.basename(local_path), repo_id=repo_id, repo_type=repo_type, token=a.token)

def maybe_download_json(repo_id: str, path_in_repo: str, repo_type: str, local_path: str) -> bool:
    try:
        fp = hf_hub_download(repo_id=repo_id, filename=path_in_repo, repo_type=repo_type, token=api().token, local_dir=os.path.dirname(local_path), local_dir_use_symlinks=False)
        os.replace(fp, local_path)
        return True
    except Exception:
        return False

def maybe_download_file(repo_id: str, path_in_repo: str, repo_type: str, local_path: str) -> bool:
    try:
        fp = hf_hub_download(repo_id=repo_id, filename=path_in_repo, repo_type=repo_type, token=api().token, local_dir=os.path.dirname(local_path), local_dir_use_symlinks=False)
        os.replace(fp, local_path)
        return True
    except Exception:
        return False

def list_latest_checkpoint(repo_id: str) -> Optional[str]:
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="model", token=api().token)
        ckpts = sorted([f for f in files if f.startswith("checkpoint-") and f.endswith("trainer_state.json")])
        if not ckpts: return None
        return "/".join(ckpts[-1].split("/")[:1])
    except Exception:
        return None

def download_folder_prefix(repo_id: str, prefix: str, dest: str):
    a = api()
    os.makedirs(dest, exist_ok=True)
    files = a.list_repo_files(repo_id=repo_id, repo_type="model")
    for f in files:
        if not f.startswith(prefix + "/"): continue
        local = os.path.join(dest, f)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        hf_hub_download(repo_id=repo_id, filename=f, repo_type="model", local_dir=os.path.dirname(local), local_dir_use_symlinks=False, token=a.token)
