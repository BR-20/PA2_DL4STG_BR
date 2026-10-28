# AI651 Assignment 2

Question 1 is a written design in `report/q1.md`. Question 2 is the Moving MNIST experiment.

## Layout

| Path | Content |
|---|---|
| `pa2/transforms.py` | mirror, rotation and the eight element group |
| `pa2/windows.py` | window and speed rules |
| `pa2/models.py` | ConvLSTM forecaster used for models A to D |
| `pa2/reach.py` | analytic reach and impulse test |
| `pa2/train.py` | training loop with speed and symmetry augmentation |
| `pa2/metrics.py` | errors, consistency gaps, symmetrised wrapper, timing |
| `pa2/flow.py` | optical flow for Figure 1 |
| `run.py` | one command per stage |
| `predictions/` | drafts of Q2.5 and Q2.11 |
| `report/make_report.py` | builds the PDF from the saved results |
| `PA2_runbook.ipynb` | the stages in order on a Colab GPU |

## Order

1. `python run.py table1`, `flow`, `reach`
2. `python run.py train --model A --seed 0` and `--seed 1`, then `table2`
3. `python predictions/make_q25.py`, commit `predictions/Q2.5.md`
4. `python run.py evaluate --models A`, `select_videos`, `figures`
5. `python predictions/make_q211.py`, commit `predictions/Q2.11.md`
6. `python run.py select_d --epochs 10`, then train B, C and D with both seeds
7. `python run.py evaluate --models A B C D`, `select_videos`, `figures`, `diagnostics`
8. `python report/make_report.py --q2 --final D --repo <url>`

Tests run with `python tests/test_all.py`.

GIFs are saved in `gifs/`.
