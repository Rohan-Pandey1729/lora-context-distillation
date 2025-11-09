import os, yaml, time, json, pathlib, requests

def load_conf():
    with open("conf/config.yaml","r") as f:
        cfg = yaml.safe_load(f)
    run_id = os.environ.get("RUN_ID","default-run")
    user = cfg["hf_username"]
    repos = {k: v.format(user=user, run_id=run_id) for k,v in cfg["repos"].items()}
    cfg["run_id"] = run_id
    cfg["repos_fmt"] = repos
    return cfg

def wait_vllm_ready(port:int, timeout_s:int=600):
    url = f"http://127.0.0.1:{port}/v1/models"
    t0 = time.time()
    while time.time()-t0 < timeout_s:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("vLLM server not ready")

def json_load(path):
    p = pathlib.Path(path)
    if not p.exists(): return {}
    with open(p,"r") as f: return json.load(f)

def json_dump(obj, path):
    import json, tempfile
    p=os.path.dirname(path); 
    if p: os.makedirs(p, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp,"w") as f: json.dump(obj,f,indent=2)
    os.replace(tmp, path)
