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

import time, requests

def wait_vllm_ready(port: int, timeout_s: int = 900, api_key: str | None = None):
    base = f"http://127.0.0.1:{port}"

    # Avoid env proxy surprises
    sess = requests.Session()
    sess.trust_env = False

    t0 = time.time()
    last = None

    while time.time() - t0 < timeout_s:
        try:
            # 1) Check server is up (no /v1 auth)
            r = sess.get(f"{base}/health", timeout=5)
            if r.status_code == 200:
                r2 = sess.get(f"{base}/v1/models", headers=headers, timeout=10)
        except Exception as e:
            last = repr(e)

        time.sleep(2)
    print(r)
    print(r2)
    raise TimeoutError(f"vLLM not ready after {timeout_s}s, last={last}")

    

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
