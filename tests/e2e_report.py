import os, shutil, subprocess, sys, types
tmp = sys.argv[1]
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root); sys.path.insert(0, os.path.dirname(__file__))
from synthetic import make_videos
import run
data = make_videos(220, seed=5)
run.get_data = lambda: (data[:120], data[120:150], data[150:220])
os.chdir(tmp)
for d in ("report", "predictions", "pa2"):
    if not os.path.exists(d):
        shutil.copytree(os.path.join(root, d), d)
shutil.copy(os.path.join(root, "report", "make_report.py"), "report/make_report.py")
run.stage_table1(None)
run.stage_flow(None)
run.stage_diagnostics(types.SimpleNamespace(final="D"))
env = dict(os.environ, PA2_ROOT=tmp, PYTHONPATH=tmp)
for s in ("make_q25.py", "make_q211.py"):
    print(subprocess.run([sys.executable, os.path.join("predictions", s)], env=env, capture_output=True, text=True, cwd=tmp))
print(subprocess.run([sys.executable, "report/make_report.py", "--q2", "--final", "D", "--repo", "https://example.org/r", "--out", os.path.join(tmp, "report", "PA2_report")],
                     env=env, capture_output=True, text=True, cwd=tmp))
