import os, sys, tempfile, types
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root); sys.path.insert(0, os.path.dirname(__file__))
from synthetic import make_videos
import run

data = make_videos(220, seed=5)
train, val, test = data[:120], data[120:150], data[150:220]
run.get_data = lambda: (train, val, test)
tmp = tempfile.mkdtemp()
os.chdir(tmp)
ns = lambda **kw: types.SimpleNamespace(**kw)
run.stage_table1(None); run.stage_reach(None)
for m in ("A", "B", "C", "D"):
    for s in (0, 1):
        run.stage_train(ns(model=m, seed=s, epochs=1))
run.stage_select_d(ns(epochs=1))
print(open("results/d_selection.json").read()[:300])
run.stage_evaluate(ns(models=["A", "B", "C", "D"]))
run.stage_select_videos(None)
print(open("results/selection.json").read())
run.stage_figures(ns(final="D"))
print(sorted(os.listdir("figures")), sorted(os.listdir("gifs")))
print("E2E OK", tmp)
