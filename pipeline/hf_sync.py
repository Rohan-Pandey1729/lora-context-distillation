import os
import pathlib
import re
from typing import Optional, List
from huggingface_hub import (
    HfApi,
    hf_hub_download,
    upload_file,
    upload_folder,
    create_repo,
    list_repo_files,
)

def api() -> HfApi:
    tok = os.environ.get("HF_TOKEN") or (
        pathlib.Path("secrets/hf_token").read_text().strip()
        if pathlib.Path("secrets/hf_token").exists() else ""
    )
    if not tok:
        raise RuntimeError("HF_TOKEN is required for all HF Hub operations")
    return HfApi(token=tok)

def ensure_repo(repo_id: str, repo_type: str) -> None:
    a = api()
    create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=False, token=a.token)

def upload_path(repo_id: str, local_path: str, repo_type: str) -> None:
    a = api()
    if os.path.isdir(local_path):
        upload_folder(
            path=local_path,
            repo_id=repo_id,
            repo_type=repo_type,
            token=a.token,
            ignore_patterns=[".git/*", "**/*.pt", "**/*.bin"],
        )
    else:
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=os.path.basename(local_path),
            repo_id=repo_id,
            repo_type=repo_type,
            token=a.token,
        )

def maybe_download_file(repo_id: str, path_in_repo: str, repo_type: str, local_path: str) -> bool:
    try:
        fp = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type=repo_type,
            token=api().token,
            local_dir=os.path.dirname(local_path),
            local_dir_use_symlinks=False,
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        os.replace(fp, local_path)
        return True
    except Exception:
        return False

def maybe_download_json(repo_id: str, path_in_repo: str, repo_type: str, local_path: str) -> bool:
    return maybe_download_file(repo_id, path_in_repo, repo_type, local_path)

def _checkpoint_prefixes(files: List[str]) -> List[str]:
    prefixes = set()
    for f in files:
        parts = f.split("/")
        if parts and parts[0].startswith("checkpoint-"):
            prefixes.add(parts[0])
    return sorted(prefixes, key=lambda s: int(re.findall(r"\d+", s)[0]) if re.findall(r"\d+", s) else -1)

def list_latest_checkpoint(repo_id: str) -> Optional[str]:
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="model", token=api().token)
        prefixes = _checkpoint_prefixes(files)
        return prefixes[-1] if prefixes else None
    except Exception:
        return None

def download_folder_prefix(repo_id: str, prefix: str, local_dir: str) -> None:
    files = list_repo_files(repo_id=repo_id, repo_type="model", token=api().token)
    wanted = [f for f in files if f.startswith(prefix + "/")]
    for f in wanted:
        dst = os.path.join(local_dir, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        fp = hf_hub_download(
            repo_id=repo_id,
            filename=f,
            repo_type="model",
            token=api().token,
            local_dir=os.path.dirname(dst),
            local_dir_use_symlinks=False,
        )
        os.replace(fp, dst)
