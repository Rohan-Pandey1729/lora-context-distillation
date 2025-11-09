import os, yaml, subprocess, tempfile
from pipeline.util import load_conf
def main():
    cfg = load_conf()
    run_id = cfg["run_id"]
    new_a = os.path.abspath("runs/A_final")
    assert os.path.isdir(new_a), "runs/A_final missing"
    with open("conf/mk_apply_template.yml","r") as f:
        mk = yaml.safe_load(f)
    for m in mk["models"]:
        if m["model"] == "__NEW_A__":
            m["model"] = new_a
    with tempfile.TemporaryDirectory() as td:
        y = os.path.join(td, "mk.yml")
        import yaml as yy
        with open(y,"w") as g: yy.safe_dump(mk, g)
        out = f"runs/{run_id}/B_new"
        os.makedirs(out, exist_ok=True)
        subprocess.check_call(["mergekit-yaml", y, out])
    print("New B saved at", out)
if __name__ == "__main__":
    main()
